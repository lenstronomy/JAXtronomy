==========
JAXtronomy
==========

.. image:: https://github.com/lenstronomy/JAXtronomy/workflows/Tests/badge.svg
    :target: https://github.com/lenstronomy/JAXtronomy/actions

.. image:: https://img.shields.io/badge/License-BSD_3--Clause-blue.svg
    :target: https://github.com/lenstronomy/lenstronomy/blob/main/LICENSE

.. image:: https://codecov.io/gh/lenstronomy/JAXtronomy/graph/badge.svg?token=6EJAX8CF62 
    :target: https://codecov.io/gh/lenstronomy/JAXtronomy

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. image:: https://img.shields.io/badge/%20formatter-docformatter-fedcba.svg
    :target: https://github.com/PyCQA/docformatter

.. image:: https://img.shields.io/badge/%20style-sphinx-0a507a.svg
    :target: https://www.sphinx-doc.org/en/master/usage/index.html

.. image:: https://img.shields.io/pypi/v/jaxtronomy?label=PyPI&logo=pypi
    :target: https://pypi.python.org/pypi/jaxtronomy

**JAX port of lenstronomy, for parallelized, GPU accelerated, and differentiable gravitational lensing and image simulations.**

**Disclaimer**: This project is still in an early development phase and serves as a skeleton for someone taking the lead on it :)

The goal of this library is to reimplement lenstronomy functionalities in pure JAX to allow for automatic differentiation, GPU acceleration, and batched computations.

**Guiding Principles**:

- Strive to be a drop-in replacement for lenstronomy, i.e. provide a close match to the lenstronomy API.
- Each function/feature will be tested against the reference lenstronomy implementation.
- This package will aim to be a **subset** of lenstronomy (i.e. only contains functions with a reference lenstronomy implementation).
- Implementations should be easy to read and understand.
- Code should be pip installable on any machine, no compilation required.
- Any notable differences between the JAX and reference implementations will be clearly documented.

Performance comparison between jaxtronomy and lenstronomy
---------------------------------------------------------

We compare the runtimes between jaxtronomy and lenstronomy by timing 10,000 function executions. These tests were done on one CPU. We expect the performance boosts to be even higher on GPU. A notebook for runtime comparisons is provided.

**LensModel ray-shooting**

The table below shows how much faster jaxtronomy is compared to lenstronomy for different deflector profiles and different grid sizes.

.. list-table::
   :header-rows: 1

   * - Deflector Profile
     - 60x60 grid
     - 180x180 grid
   * - CONVERGENCE
     - 1.5x
     - 6.9x
   * - CSE
     - 5.7x
     - 10.1x
   * - EPL (jax) vs EPL_NUMBA
     - 1.3x
     - 2.2x
   * - EPL_Q_PHI
     - 0.3x
     - 0.4x
   * - GAUSSIAN
     - 2.1x
     - 3.3x
   * - GAUSSIAN_POTENTIAL
     - 1.9x
     - 3.1x
   * - HERNQUIST
     - 1.6x
     - 2.7x
   * - HERNQUIST_ELLIPSE_CSE
     - 4.7x
     - 5.9x
   * - LOS
     - 2.9x
     - 7.6x
   * - LOS_MINIMAL
     - 2.8x
     - 8.1x
   * - NFW
     - 2.0x
     - 4.0x
   * - NFW_ELLIPSE_CSE
     - 5.4x
     - 7.0x
   * - NIE
     - 1.4x
     - 1.9x
   * - PJAFFE
     - 2.4x
     - 2.4x
   * - PJAFFE_ELLIPSE_POTENTIAL
     - 3.1x
     - 3.1x
   * - SHEAR
     - 2.1x
     - 5.3x
   * - SIE
     - 1.2x
     - 1.9x
   * - SIS
     - 3.7x
     - 4.3x
   * - SPP
     - 1.4x
     - 2.3x

**LightModel surface brightness**

The table below shows how much faster jaxtronomy is compared to lenstronomy for different source profiles and different grid sizes.

.. list-table::
   :header-rows: 1

   * - Source Profile
     - 60x60 grid
     - 180x180 grid
   * - CORE_SERSIC
     - 4.4x
     - 14.8x
   * - GAUSSIAN
     - 3.8x
     - 9.9x
   * - GAUSSIAN_ELLIPSE
     - 2.7x
     - 7.8x
   * - MULTI_GAUSSIAN (5 components)
     - 8.0x
     - 18.2x
   * - MULTI_GAUSSIAN_ELLIPSE (5 components)
     - 8.2x
     - 18.1x
   * - SERSIC
     - 3.4x
     - 10.0x
   * - SERSIC_ELLIPSE
     - 3.5x
     - 9.7x
   * - SERSIC_ELLIPSE_Q_PHI
     - 4.0x
     - 9.6x

**Image Convolution**

There is no gaussian convolution function in the JAX library. Thus, in jaxtronomy we construct a gaussian pixel kernel, pad the image, and perform an fft convolution which mimics scipy.ndimage.gaussian_filter with mode="nearest".
When the kernel radius is less than 10, jaxtronomy takes about 1.1x to 1.2x longer than lenstronomy to perform a gaussian convolution, and when the kernel size is larger, it takes jaxtronomy 2x as long or more.
Further details can be found in the performance comparison notebook.

Related software packages
-------------------------

The following lensing software packages do use JAX-accelerated computing that in part were inspired or made use of lenstronomy functions:

- Herculens_
- GIGA-lens_
- PaltaX_

.. _Herculens: https://github.com/herculens/herculens
.. _GIGA-lens: https://github.com/giga-lens/gigalens
.. _PaltaX: https://github.com/swagnercarena/paltax





