# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Handling surfaces
-----------------
"""
import os
import numpy as np
import nibabel as nb

from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    BaseInterfaceInputSpec, TraitedSpec, File, isdefined,
    SimpleInterface, traits
)


class NormalizeSurfInputSpec(BaseInterfaceInputSpec):
    in_file = File(mandatory=True, exists=True, desc='Freesurfer-generated GIFTI file')
    transform_file = File(exists=True, desc='FSL or LTA affine transform file')


class NormalizeSurfOutputSpec(TraitedSpec):
    out_file = File(desc='output file with re-centered GIFTI coordinates')


class NormalizeSurf(SimpleInterface):
    """ Normalizes a FreeSurfer-generated GIFTI image
    FreeSurfer includes an offset to the center of the brain volume that is not
    respected by all software packages.
    Normalization involves adding this offset to the coordinates of all
    vertices, and zeroing out that offset, to ensure consistent behavior
    across software packages.
    In particular, this normalization is consistent with the Human Connectome
    Project pipeline (see `AlgorithmSurfaceApplyAffine`_ and
    `FreeSurfer2CaretConvertAndRegisterNonlinear`_), although the the HCP
    may not zero out the offset.
    GIFTI files with ``midthickness``/``graymid`` in the name are also updated
    to include the following metadata entries::
        {
            AnatomicalStructureSecondary: MidThickness,
            GeometricType: Anatomical
        }
    This interface is intended to be applied uniformly to GIFTI surface files
    generated from the ``?h.white``/``?h.smoothwm`` and ``?h.pial`` surfaces,
    as well as externally-generated ``?h.midthickness``/``?h.graymid`` files.
    In principle, this should apply safely to any other surface, although it is
    less relevant to surfaces that don't describe an anatomical structure.
    .. _AlgorithmSurfaceApplyAffine: https://github.com/Washington-University/workbench/blob/1b79e56/src/Algorithms/AlgorithmSurfaceApplyAffine.cxx#L73-L91
    .. _FreeSurfer2CaretConvertAndRegisterNonlinear: https://github.com/Washington-University/Pipelines/blob/ae69b9a/PostFreeSurfer/scripts/FreeSurfer2CaretConvertAndRegisterNonlinear.sh#L147-154
    """
    input_spec = NormalizeSurfInputSpec
    output_spec = NormalizeSurfOutputSpec

    def _run_interface(self, runtime):
        transform_file = self.inputs.transform_file
        if not isdefined(transform_file):
            transform_file = None
        self._results['out_file'] = normalize_surfs(self.inputs.in_file, transform_file)
        return runtime


class ApplyLTATransformInputSpec(BaseInterfaceInputSpec):
    in_file = File(mandatory=True, exists=True, desc='Freesurfer-generated GIFTI file')
    transform_file = File(mandatory=True, exists=True, desc='LTA affine transform file')
    absolute_coords = traits.Bool(True, usedefault=True, desc='coordinates are absolute')


class ApplyLTATransformOutputSpec(TraitedSpec):
    out_file = File(desc='output file with re-centered GIFTI coordinates')


class ApplyLTATransform(SimpleInterface):
    """
    Apply an affine transform (given in LTA format) to a GIFTI surface.
    """
    input_spec = ApplyLTATransformInputSpec
    output_spec = ApplyLTATransformOutputSpec

    def _run_interface(self, runtime):
        self._results['out_file'] = surf2surf(
            self.inputs.in_file,
            self.inputs.transform_file,
            newpath=runtime.cwd,
            center=self.inputs.absolute_coords)
        return runtime


def load_transform(fname):
    """Load affine transform from file
    Parameters
    ----------
    fname : str or None
        Filename of an LTA or FSL-style MAT transform file.
        If ``None``, return an identity transform
    Returns
    -------
    affine : (4, 4) numpy.ndarray
    """
    if fname is None:
        return np.eye(4)

    if fname.endswith('.mat'):
        return np.loadtxt(fname)
    elif fname.endswith('.lta'):
        with open(fname, 'rb') as fobj:
            for line in fobj:
                if line.startswith(b'1 4 4'):
                    break
            lines = fobj.readlines()[:4]
        return np.genfromtxt(lines)

    raise ValueError("Unknown transform type; pass FSL (.mat) or LTA (.lta)")


def normalize_surfs(in_file, transform_file):
    """ Re-center GIFTI coordinates to fit align to native T1 space
    For midthickness surfaces, add MidThickness metadata
    Coordinate update based on:
    https://github.com/Washington-University/workbench/blob/1b79e56/src/Algorithms/AlgorithmSurfaceApplyAffine.cxx#L73-L91
    and
    https://github.com/Washington-University/Pipelines/blob/ae69b9a/PostFreeSurfer/scripts/FreeSurfer2CaretConvertAndRegisterNonlinear.sh#L147
    """

    img = nb.load(in_file)
    transform = load_transform(transform_file)
    pointset = img.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0]
    coords = pointset.data.T
    c_ras_keys = ('VolGeomC_R', 'VolGeomC_A', 'VolGeomC_S')
    ras = np.array([[float(pointset.metadata.get(key, 0.0))]
                    for key in c_ras_keys])
    ones = np.ones((1, coords.shape[1]), dtype=coords.dtype)
    # Apply C_RAS translation to coordinates, then transform
    pointset.data = transform.dot(np.vstack((coords + ras, ones)))[:3].T.astype(coords.dtype)

    secondary = nb.gifti.GiftiNVPairs('AnatomicalStructureSecondary', 'MidThickness')
    geom_type = nb.gifti.GiftiNVPairs('GeometricType', 'Anatomical')
    has_ass = has_geo = False
    for nvpair in pointset.meta.data:
        # Remove C_RAS translation from metadata to avoid double-dipping in FreeSurfer
        if nvpair.name in c_ras_keys:
            nvpair.value = '0.000000'
        # Check for missing metadata
        elif nvpair.name == secondary.name:
            has_ass = True
        elif nvpair.name == geom_type.name:
            has_geo = True
    fname = os.path.basename(in_file)
    # Update metadata for MidThickness/graymid surfaces
    if 'midthickness' in fname.lower() or 'graymid' in fname.lower():
        if not has_ass:
            pointset.meta.data.insert(1, secondary)
        if not has_geo:
            pointset.meta.data.insert(2, geom_type)
    img.to_filename(fname)
    return os.path.abspath(fname)


def surf2surf(in_surf, xform_lta, invert=True, newpath=None,
              center=False,
              offset_from=['VolGeomC_R', 'VolGeomC_A', 'VolGeomC_S']):
    """
    Transforms a gifti surface onto a target space
    """
    # Read gifti surface
    surfgii = nb.load(in_surf)

    # Read in lta file
    with open(xform_lta) as xfmfile:
        xfm = np.array([np.fromstring(l, sep=" ").tolist()
                       for l in xfmfile.readlines()[5:9]])

    if invert:
        xfm = np.linalg.inv(xfm)

    coords = surfgii.darrays[0].data
    newcoords = xfm.dot(np.hstack((
        coords, np.ones((coords.shape[0], 1)))).T).T[..., :3]

    meta = surfgii.darrays[0].meta.metadata

    coordsys = surfgii.darrays[0].coordsys
    coordsys.dataspace = 2
    coordsys.xformspace = 2
    coordsys.xform = np.eye(4)

    # Absolute coordinates
    if center:
        offset = [float(surfgii.darrays[0].meta.metadata[v]) for v in offset_from]
        newcoords += offset
        for k in offset_from:
            meta[k] = '%f' % 0.0

    # Updating metadata
    # del meta['SurfaceCenterX']
    # del meta['SurfaceCenterY']
    # del meta['SurfaceCenterZ']

    surfgii.darrays[0] = nb.gifti.GiftiDataArray(
        data=newcoords.astype('float32'),
        datatype='NIFTI_TYPE_FLOAT32',
        intent='NIFTI_INTENT_POINTSET',
        meta=meta,
        coordsys=coordsys
    )
    surfgii.darrays[1] = nb.gifti.GiftiDataArray(
        data=surfgii.darrays[1].data.astype('float32'),
        datatype='NIFTI_TYPE_FLOAT32',
        intent='NIFTI_INTENT_TRIANGLE',
        coordsys=None,
        # meta=None,
        # coordsys=nb.gifti.GiftiCoordSystem(dataspace='NIFTI_XFORM_UNKNOWN'),
    )

    out_file = fname_presuffix(in_surf,
                               suffix='_target.surf.gii',
                               use_ext=False,
                               newpath=newpath)
    surfgii.to_filename(out_file)
    return out_file
