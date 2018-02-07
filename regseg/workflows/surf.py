# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Surface extraction
++++++++++++++++++

Defines the workflows for extracting surfaces from segmentations


"""
import nipype.pipeline.engine as pe             # pipeline engine
from nipype.interfaces import utility as niu    # utility
from nipype.interfaces import io as nio         # i/o
from nipype.interfaces import freesurfer as fs  # Freesurfer
from ..interfaces import Binarize, NormalizeSurf, FillMask, AsegAddOuter


def extract_surfs_fs_wf(name='extract_surfs_fs_wf'):
    """
    This workflow extracts GIFTI sorfaces from a FreeSurfer subjects directory
    and projects them onto a target space.

    .. workflow::
        :graph2use: orig
        :simple_form: yes
        from regseg.workflows.surf import extract_surfs_fs_wf
        wf = extract_surfs_fs_wf()
    **Inputs**
        subjects_dir
            FreeSurfer SUBJECTS_DIR
        subject_id
            FreeSurfer subject ID
        t1_preproc
            The T1w preprocessed image
        in_target
            A target space (B0 for dMRI or reference for BOLD)
        target_to_t1_lta
            A target-to-T1w affine transform, in LTA format.

    **Outputs**
        surfaces
            GIFTI surfaces
    """
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(['subjects_dir', 'subject_id',
                                               't1_preproc', 'in_target']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(['surfaces']), name='outputnode')

    get_surfaces = pe.Node(nio.FreeSurferSource(), name='get_surfaces')

    exsurfs = extract_surfaces(normalize=False)
    exsurfs.inputs.inputnode.model_name = 'bold'

    workflow.connect([
        (inputnode, get_surfaces, [('subjects_dir', 'subjects_dir'),
                                   ('subject_id', 'subject_id')]),
        (get_surfaces, exsurfs, [('aseg', 'inputnode.aseg'),
                                 ('norm', 'inputnode.norm')])
    ])

    return workflow

def extract_surfaces(name='GenSurface', normalize=True):
    """
    A nipype workflow for surface extraction from ``labels`` in a segmentation.

    .. note :: References used to implement this code:

        * <https://github.com/nipy/nipype/issues/307>
        * <https://mail.nmr.mgh.harvard.edu/pipermail/\
freesurfer/2011-November/021391.html>
        * <http://brainder.org/2012/05/08/importing-\
freesurfer-subcortical-structures-into-blender/>
        * <https://mail.nmr.mgh.harvard.edu/pipermail/\
freesurfer/2013-June/030586.html>
    """
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['aseg', 'norm', 't1_2_fsnative_invxfm', 'in_filled', 'model_name'],
        mandatory_inputs=False),
        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_surf', 'out_binary']), name='outputnode')

    get_mod = pe.Node(niu.Function(function=_read_model, output_names=['name', 'labels']),
                      name='GetModel')

    binarize = pe.MapNode(Binarize(), name='BinarizeLabels',
                          iterfield=['match'])
    fill = pe.MapNode(FillMask(), name='FillMask', iterfield=['in_file'])
    pretess = pe.MapNode(fs.MRIPretess(label=1), name='PreTess',
                         iterfield=['in_filled'])
    tess = pe.MapNode(fs.MRITessellate(label_value=1, use_real_RAS_coordinates=True),
                      name='tess', iterfield=['in_file'])
    smooth = pe.MapNode(fs.SmoothTessellation(disable_estimates=True),
                        name='mris_smooth', iterfield=['in_file'])
    rename = pe.MapNode(niu.Rename(keep_ext=False),
                        name='rename', iterfield=['in_file', 'format_string'])

    togii = pe.MapNode(fs.MRIsConvert(out_datatype='gii'),
                       iterfield='in_file', name='toGIFTI')

    wf = pe.Workflow(name=name)
    wf.connect([
        (inputnode, get_mod, [('model_name', 'model_name')]),
        (inputnode, binarize, [('aseg', 'in_file')]),
        (get_mod, binarize, [('labels', 'match')]),
        (inputnode, pretess, [('norm', 'in_norm')]),
        (inputnode, fill, [('in_filled', 'in_filled')]),
        (binarize, fill, [('out_file', 'in_file')]),
        (fill, pretess, [('out_file', 'in_filled')]),
        (pretess, tess, [('out_file', 'in_file')]),
        (tess, smooth, [('surface', 'in_file')]),
        (smooth, rename, [('surface', 'in_file')]),
        (get_mod, rename, [('name', 'format_string')]),
        (rename, togii, [('out_file', 'in_file')]),
        (fill, outputnode, [('out_file', 'out_binary')]),
    ])
    if normalize:
        fixgii = pe.MapNode(NormalizeSurf(), iterfield='in_file', name='fixGIFTI')
        wf.connect([
            (inputnode, fixgii, [('t1_2_fsnative_invxfm', 'transform_file')]),
            (togii, fixgii, [('converted', 'in_file')]),
            (fixgii, outputnode, [('out_file', 'out_surf')]),
        ])
    else:
        wf.connect([
            (togii, outputnode, [('converted', 'out_surf')]),
        ])

    return wf


def extract_surfaces_model(model='bold', name='Surfaces', gen_outer=True):
    """Extracts surfaces as prescribed by the model ``model``"""

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['aseg', 'norm', 'brainmask', 't1_2_fsnative_invxfm']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_surf']), name='outputnode')

    exsurfs = extract_surfaces()
    exsurfs.inputs.inputnode.model_name = model

    wf = pe.Workflow(name=name)
    wf.connect([
        (inputnode, exsurfs, [('norm', 'inputnode.norm'),
                              ('t1_2_fsnative_invxfm', 'inputnode.t1_2_fsnative_invxfm')]),
        (exsurfs, outputnode, [('outputnode.out_surf', 'out_surf')]),
    ])

    if gen_outer:
        addmsk = pe.Node(AsegAddOuter(), name='addmsk')
        wf.connect([
            (inputnode, addmsk, [('aseg', 'in_file'),
                                 ('brainmask', 'in_mask')]),
            (addmsk, exsurfs, [('out_file', 'inputnode.aseg')]),
        ])
    else:
        wf.connect([
            (inputnode, exsurfs, [('aseg', 'inputnode.aseg')]),
        ])

    return wf


def _read_model(model_name):
    from sys import version_info
    import simplejson as json
    from pkg_resources import resource_filename as pkgrf

    with open(pkgrf('regseg', 'data/model_%s.json' % model_name.lower()),
              'rb' if version_info[0] < 3 else 'r') as sfh:
        model = json.load(sfh)

    name = ['%s.surf' % m for m in model.keys()]
    labels = list(model.values())

    return name, labels
