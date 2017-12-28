# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""

Base interface for image registration

"""

from sklearn.base import BaseEstimator, TransformerMixin
from abc import ABCMeta, abstractmethod
import six


class _BaseRegistration(six.with_metaclass(ABCMeta, BaseEstimator, TransformerMixin)):
    """
    A base, abstract class for image registration methods
    """

    @abstractmethod
    def fit(fixed, moving):
        """
        Placeholder for fit. Subclasses should implement this method!

        Perform registration of moving into fixed

        Parameters
        ----------

        fixed : str or nibabel image obj
                Registration-target image

        moving : str or nibabel surface or image object
                 The moving object

        Returns
        -------

        self : object
            Returns the instance itself.

        """

    @abstractmethod
    def transform(self, moving, fixed=None):
        """
        Apply transform to the moving object.

        ``moving`` is projected into the target ``fixed`` space.


        """

    @abstractmethod
    def _compute_gradient(self):
        """
        Returns the gradient with new parameters

        """
