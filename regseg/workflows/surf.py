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
from ..interfaces import (
    Binarize, NormalizeSurf, FillMask, AsegAddOuter, ApplyLTATransform
)


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
            The T1w preprocessed image (the co-registration target for
            bbr/bbregister)
        target_to_t1_lta
            A target-to-T1w affine transform, in LTA format.

    **Outputs**
        out_surf
            GIFTI surfaces, in target space
    """
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface([
        'subjects_dir', 'subject_id', 't1_preproc', 'xform_trg2t1']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(['out_surf']), name='outputnode')

    get_fs = pe.Node(nio.FreeSurferSource(), name='get_fs')
    exsurfs = extract_surfaces(normalize=False, use_ras_coord=False, brainmask=True)
    exsurfs.inputs.inputnode.model_name = 'boldsimple'

    tkreg = pe.Node(fs.Tkregister2(reg_file='native2fs.dat', noedit=True,
                    reg_header=True), name='tkregister2')

    def _format_subid(sub_id):
        return '--subject %s' % sub_id
    lta_conv = pe.Node(fs.utils.LTAConvert(out_lta=True), 'lta_convert')
    lta_concat = pe.Node(fs.preprocess.ConcatenateLTA(out_type='RAS2RAS'), name='lta_concat')
    lta_xfm = pe.MapNode(ApplyLTATransform(), iterfield=['in_file'], name='lta_xfm')

    workflow.connect([
        (inputnode, get_fs, [('subjects_dir', 'subjects_dir'),
                             ('subject_id', 'subject_id')]),
        (inputnode, tkreg, [('t1_preproc', 'moving_image'),
                            ('subject_id', 'subject_id')]),
        (inputnode, lta_conv, [('t1_preproc', 'source_file'),
                               (('subject_id', _format_subid), 'args')]),
        (inputnode, lta_concat, [('subjects_dir', 'subjects_dir'),
                                 ('subject_id', 'subject'),
                                 ('xform_trg2t1', 'in_lta1')]),
        (get_fs, exsurfs, [('aseg', 'inputnode.aseg'),
                           ('norm', 'inputnode.norm'),
                           ('brainmask', 'inputnode.brainmask')]),
        (get_fs, tkreg, [('orig', 'target_image')]),
        (tkreg, lta_conv, [('reg_file', 'in_reg')]),
        (get_fs, lta_conv, [('orig', 'target_file')]),
        (lta_conv, lta_concat, [('out_lta', 'in_lta2')]),
        (lta_concat, lta_xfm, [('out_file', 'transform_file')]),
        (exsurfs, lta_xfm, [('outputnode.out_surf', 'in_file')]),
        (lta_xfm, outputnode, [('out_file', 'out_surf')]),
    ])

    return workflow


def mask2surf(name='MaskToSurface', use_ras_coord=True):
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_file', 'norm', 'in_filled', 'out_name']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_surf']), name='outputnode')
    fill = pe.Node(FillMask(), name='FillMask')
    pretess = pe.Node(fs.MRIPretess(label=1), name='PreTess')
    tess = pe.Node(fs.MRITessellate(label_value=1, use_real_RAS_coordinates=use_ras_coord),
                   name='tess')
    smooth = pe.Node(fs.SmoothTessellation(disable_estimates=True),
                     name='mris_smooth')
    rename = pe.Node(niu.Rename(keep_ext=False), name='rename')
    togii = pe.Node(fs.MRIsConvert(out_datatype='gii'), name='toGIFTI')

    wf = pe.Workflow(name=name)
    wf.connect([
        (inputnode, pretess, [('norm', 'in_norm')]),
        (inputnode, fill, [('in_file', 'in_file'),
                           ('in_filled', 'in_filled')]),
        (inputnode, rename, [('out_name', 'format_string')]),
        (fill, pretess, [('out_file', 'in_filled')]),
        (pretess, tess, [('out_file', 'in_file')]),
        (tess, smooth, [('surface', 'in_file')]),
        (smooth, rename, [('surface', 'in_file')]),
        (rename, togii, [('out_file', 'in_file')]),
        (togii, outputnode, [('converted', 'out_surf')]),
    ])
    return wf


def extract_surfaces(name='GenSurface', normalize=True, use_ras_coord=True,
                     brainmask=False):
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
        fields=['aseg', 'norm', 'in_filled', 'brainmask', 't1_2_fsnative_invxfm', 'model_name'],
        mandatory_inputs=False),
        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_surf', 'out_binary']), name='outputnode')

    surfnode = pe.Node(niu.IdentityInterface(fields=['out_surf']), name='surfnode')

    get_mod = pe.Node(niu.Function(function=_read_model, output_names=['name', 'labels']),
                      name='GetModel')

    binarize = pe.MapNode(Binarize(), name='BinarizeLabels',
                          iterfield=['match'])

    fill = pe.MapNode(FillMask(), name='FillMask', iterfield=['in_file'])
    pretess = pe.MapNode(fs.MRIPretess(label=1), name='PreTess',
                         iterfield=['in_filled'])
    tess = pe.MapNode(fs.MRITessellate(label_value=1, use_real_RAS_coordinates=use_ras_coord),
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

    if brainmask:
        bmsk_wf = mask2surf(use_ras_coord=use_ras_coord)
        bmsk_wf.inputs.inputnode.out_name = 'brain.surf'
        merge = pe.Node(niu.Merge(2), name='mergebmask')
        wf.connect([
            (inputnode, bmsk_wf, [('brainmask', 'inputnode.in_file'),
                                  ('norm', 'inputnode.norm'),
                                  ('in_filled', 'inputnode.in_filled')]),
            (togii, merge, [('converted', 'in1')]),
            (bmsk_wf, merge, [('outputnode.out_surf', 'in2')]),
            (merge, surfnode, [('out', 'out_surf')]),
        ])
    else:
        wf.connect(togii, 'converted', surfnode, 'out_surf')

    if normalize:
        fixgii = pe.MapNode(NormalizeSurf(), iterfield='in_file', name='fixGIFTI')
        wf.connect([
            (inputnode, fixgii, [('t1_2_fsnative_invxfm', 'transform_file')]),
            (surfnode, fixgii, [('out_surf', 'in_file')]),
            (fixgii, outputnode, [('out_file', 'out_surf')]),
        ])
    else:
        wf.connect([
            (surfnode, outputnode, [('out_surf', 'out_surf')]),
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

