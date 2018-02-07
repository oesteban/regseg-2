# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
RegSeg2
"""

from datetime import date
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

PACKAGE_NAME = 'regseg'

__author__ = 'Oscar Esteban'
__email__ = 'code@oscaresteban.es'
__maintainer__ = 'Oscar Esteban'
__copyright__ = ('Copyright %d, %s and Center for Reproducible Neuroscience, '
                 'Stanford University') % (date.today().year, __author__)
__credits__ = __author__
__license__ = 'MIT License'
__status__ = '3 - Alpha'
__description__ = 'Surface-driven 3D image registration in python'
__longdesc__ = ('RegSeg is an image joint segmentation-registration method that '
                'maps surfaces into volumetric, multivariate 3D data. The surfaces '
                'must be triangular meshes, and they drive the registration process '
                'in a way that the parametric properties of the regions defined by '
                'the surfaces are best fitted.')

__url__ = 'http://{}.readthedocs.org/'.format(PACKAGE_NAME)
__download__ = ('https://github.com/oesteban/{}-2/archive/'
                '{}.tar.gz').format(PACKAGE_NAME, __version__)


CLASSIFIERS = [
    'Development Status :: %s' % __status__,
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Image Recognition',
    'License :: OSI Approved :: %s' % __license__,
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
]

SETUP_REQUIRES = []

REQUIRES = [
    'numpy>=1.12.0',
    'scikit-learn>=0.19.0',
    'scipy',
    'six',
    'nibabel',
    'versioneer',
]

LINKS_REQUIRES = [
]


TESTS_REQUIRES = [
    'pytest',
    'codecov',
    'pytest-xdist',
]

EXTRA_REQUIRES = {
    'doc': ['sphinx>=1.5,<1.6', 'sphinx_rtd_theme>=0.2.4', 'sphinx-argparse'],
    'tests': TESTS_REQUIRES,
    'notebooks': ['ipython', 'jupyter'],
}

# Enable a handle to install all extra dependencies at once
EXTRA_REQUIRES['all'] = [val for _, val in list(EXTRA_REQUIRES.items())]
