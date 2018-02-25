# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Creating phantoms with shapes
-----------------------------
"""
from os import path as op
import numpy as np
import nibabel as nb
from nipype import logging
from nipype.interfaces.base import (
    SimpleInterface, traits, File, BaseInterfaceInputSpec, TraitedSpec
)

from ..misc import generate_nifti, generate_shape

iflogger = logging.getLogger('interface')


class PhantomInputSpec(BaseInterfaceInputSpec):
    shape = traits.Enum('gyrus', 'box', 'L', 'ball', usedefault=True,
                        desc='Phantom shape')
    cortex = traits.Bool(True, usedefault=True,
                         desc='Generate a crust mimicking cortical GM')
    out_file = File('phantom_model.nii.gz', usedefault=True,
                    desc='output file name')
    seed = traits.Int(0, usedefault=True,
                      desc='random seed')
    matrix_size_item_trait = traits.Int(61, usedefault=True)
    matrix_size = traits.Either(matrix_size_item_trait,
                                traits.List(matrix_size_item_trait),
                                default=61, usedefault=True,
                                desc='image matrix')


class PhantomOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output file name')


class Phantom(SimpleInterface):

    """
    Returns a phantom model
    """
    input_spec = PhantomInputSpec
    output_spec = PhantomOutputSpec

    def _run_interface(self, runtime):
        size = np.atleast_1d(self.inputs.matrix_size).tolist()
        if len(size) == 1:
            size = [size[0]] * 3

        self._results['out_file'] = op.join(
            runtime.cwd, self.inputs.out_file)
        classes = generate_shape(self.inputs.shape, datashape=size,
                                 cortex=self.inputs.cortex)
        data = classes[1]
        if self.inputs.cortex:
            data += 2 * classes[2]
        nii = generate_nifti(data)
        nii.to_filename(self._results['out_file'])
        return runtime
