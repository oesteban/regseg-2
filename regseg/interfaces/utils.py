# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Image tools interfaces
~~~~~~~~~~~~~~~~~~~~~~
"""
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
