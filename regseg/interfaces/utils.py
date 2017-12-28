# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Image tools interfaces
~~~~~~~~~~~~~~~~~~~~~~
"""
import os
import nibabel as nb
import numpy as np

from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits, isdefined, TraitedSpec, BaseInterfaceInputSpec,
    File, SimpleInterface
)

LOGGER = logging.getLogger('interface')


class BinarizeInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True,
                   desc='input file')
    threshold = traits.Float(0.0, usedefault=True,
                             desc='threshold')
    match = traits.List(traits.Int,
                        desc='match instead of threshold')


class BinarizeOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output binary mask')


class Binarize(SimpleInterface):
    input_spec = BinarizeInputSpec
    output_spec = BinarizeOutputSpec

    def _run_interface(self, runtime):
        self._results['out_file'] = fname_presuffix(
            self.inputs.in_file, suffix='_mask', newpath=runtime.cwd)

        nii = nb.load(self.inputs.in_file)
        data = nii.get_data()
        mask = np.zeros_like(data, dtype=np.uint8)

        if isdefined(self.inputs.match) and self.inputs.match:
            for label in self.inputs.match:
                mask[data == label] = 1
        else:
            mask[data >= self.inputs.threshold] = 1

        new = nii.__class__(mask, nii.affine, nii.header)
        new.set_data_dtype(np.uint8)
        new.to_filename(self._results['out_file'])
        return runtime


class FillMaskInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input file')
    in_filled = File(exists=True, desc='input file')


class FillMaskOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output binary mask')


class FillMask(SimpleInterface):
    input_spec = FillMaskInputSpec
    output_spec = FillMaskOutputSpec

    def _run_interface(self, runtime):
        filled = None
        if isdefined(self.inputs.in_filled):
            filled = self.inputs.in_filled

        self._results['out_file'] = _fillmask(
            self.inputs.in_file,
            in_filled=filled,
            newpath=runtime.cwd
        )
        return runtime


class AsegAddOuterInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input file')
    in_mask = File(exists=True, mandatory=True, desc='input file')
    label = traits.Int(5000, usedefault=True, desc='new label')


class AsegAddOuterOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output aseg file with label added')


class AsegAddOuter(SimpleInterface):
    input_spec = AsegAddOuterInputSpec
    output_spec = AsegAddOuterOutputSpec

    def _run_interface(self, runtime):
        self._results['out_file'] = _aseg_add_outer(
            self.inputs.in_file,
            self.inputs.in_mask,
            newlabel=self.inputs.label,
            newpath=runtime.cwd
        )
        return runtime


def _fillmask(in_file, in_filled=None, newpath=None):

    if in_filled is None:
        return in_file

    if newpath is None:
        newpath = os.getcwd()

    nii = nb.load(in_file)
    data = nii.get_data()

    in_filled = np.atleast_1d(in_filled).tolist()
    for fname in in_filled:
        data = data + nb.load(fname).get_data()
    data[data > 1.0] = 1

    out_file = fname_presuffix(in_file, suffix='_filled',
                               newpath=newpath)
    newfile = nii.__class__(data, nii.affine, nii.header)
    newfile.set_data_dtype(np.uint8)
    newfile.to_filename(out_file)
    return out_file


def _aseg_add_outer(in_aseg, brainmask, newlabel=5000, newpath=None):
    asegnii = nb.load(in_aseg)
    aseg = asegnii.get_data()

    bmsk = nb.load(brainmask).get_data().astype(int)
    bmsk[bmsk > 0] = newlabel
    bmsk[bmsk < newlabel] = 0

    aseg[aseg == 0] = bmsk[aseg == 0]

    if newpath is None:
        newpath = os.getcwd()
    out_file = fname_presuffix(in_aseg, suffix='+outer', newpath=newpath)
    asegnii.__class__(aseg, asegnii.affine, asegnii.header).to_filename(out_file)
    return out_file
