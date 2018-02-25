# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
NIfTI tools
~~~~~~~~~~~
"""
import numpy as np
import nibabel as nb


def generate_nifti(data, spacing=1.0, dtype=np.uint8):
    """
    Generates a NIfTI file from the input ``data`` ndarray
    """

    shape = np.array(np.shape(data)[:3], dtype=np.float32)

    if isinstance(spacing, tuple):
        spacing = list(spacing)

    if not isinstance(spacing, list):
        spacing = [spacing] * 3

    affine = np.diag(spacing + [1.0])
    affine[:, 3] = -0.5 * affine.dot(shape.tolist() + [1.0])

    hdr = nb.Nifti1Header()
    hdr.set_xyzt_units('mm')
    if np.ndim(data) > 3:
        hdr.set_xyzt_units('mm', 'sec')

    hdr.set_slope_inter(1.0, 0.0)
    hdr.set_intent('NIFTI_INTENT_VECTOR')
    hdr['regular'] = np.array('r', dtype='|S1')

    nii = nb.Nifti1Image(data, affine, hdr)
    nii.set_data_dtype(dtype)
    nii.set_qform(affine, code=1)
    nii.set_sform(affine, code=1)
    return nii
