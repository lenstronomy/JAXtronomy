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

.. image:: https://img.shields.io/pypi/v/JAXtronomy?label=PyPI&logo=pypi
    :target: https://pypi.python.org/pypi/JAXtronomy

**JAX port of lenstronomy, for parallelized, GPU accelerated, and differentiable gravitational lensing and image simulations.**

The goal of this library is to reimplement lenstronomy functionalities in pure JAX to allow for automatic differentiation, GPU acceleration, and batched computations.

**Guiding Principles**:

- Strive to be a drop-in replacement for lenstronomy, i.e. provide a close match to the lenstronomy API.
- Each function/feature will be tested against the reference lenstronomy implementation.
- This package will aim to be a **subset** of lenstronomy (i.e. only contains functions with a reference lenstronomy implementation).
- Implementations should be easy to read and understand.
- Code should be pip installable on any machine, no compilation required.
- Any notable differences between the JAX and reference implementations will be clearly documented.

**Installation**:

``JAXtronomy`` can be installed with ::

  pip install jaxtronomy

Performance comparison between JAXtronomy and lenstronomy
---------------------------------------------------------

We compare the runtimes between JAXtronomy and lenstronomy by timing 10,000 function executions. These tests were done on one CPU. We expect the performance boosts to be even higher on GPU. A notebook for runtime comparisons is provided.

**LensModel ray-shooting**

The table below shows how much faster JAXtronomy is compared to lenstronomy for different deflector profiles and different grid sizes.

.. list-table::
   :header-rows: 1

  * - Deflector Profile
    - 60x60 grid (cpu)
    - 180x180 grid (cpu)
    - 180x180 grid (gpu)
  * - CONVERGENCE
    - 0.4x
    - 0.8x
    - 0.1x
  * - CSE
    - 1.3x
    - 3.2x
    - 0.2x
  * - EPL
    - 8.9x
    - 14.0
    - 7.3x
  * - EPL (jax) vs EPL_NUMBA
    - 1.0x
    - 1.7x
    - 1.3x
  * - EPL_Q_PHI
    - 0.2x
    - 0.3x
    - 0.2x
  * - GAUSSIAN
    - 1.0x
    - 1.9x
    - 0.2x
  * - GAUSSIAN_POTENTIAL
    - 0.9x
    - 1.8x
    - 0.1x
  * - HERNQUIST
    - 1.7x
    - 4.3x
    - 0.4x
  * - HERNQUIST_ELLIPSE_CSE
    - 5.0x
    - 4.3x
    - 5.2x
  * - NFW
    - 1.7x
    - 3.2x
    - 0.3x
  * - NFW_ELLIPSE_CSE
    - 5.4x
    - 5.2x
    - 5.7x
  * - NIE
    - 0.8x
    - 1.6x
    - 0.2x
  * - PJAFFE
    - 0.9x
    - 0.8x
    - 0.2x
  * - PJAFFE_ELLIPSE_POTENTIAL
    - 1.4x
    - 1.6x
    - 0.2x
  * - SHEAR
    - 0.9x
    - 1.6x
    - 0.1x
  * - SIE
    - 0.9x
    - 1.2x
    - 0.3x
  * - SIS
    - 1.7x
    - 1.7x
    - 0.1x
  * - SPP
    - 0.8x
    - 1.7x
    - 0.2x
  * - TNFW
    - 2.5x
    - 3.9x
    - 0.6x

Note that some profiles' runtime are dependent on function arguments. For example, the EPL profile involves performing a hyp2f1 calculation using a power series expansion.
In lenstronomy, the number of terms used depends on how quickly the series converges, whereas in JAXtronomy, the power series always involves a fixed number of terms, which is required for autodifferentiation.
In the table, the EPL comparison shows the case where lenstronomy requires the maximum number of terms, and the EPL_Q_PHI comparison shows the case where lenstronomy requires the fewest.

**LightModel surface brightness**

The table below shows how much faster JAXtronomy is compared to lenstronomy for different source profiles and different grid sizes.

.. list-table::
   :header-rows: 1

   * - Source Profile
     - 60x60 grid
     - 180x180 grid
   * - CORE_SERSIC
     - 4.4x
     - 15x
   * - GAUSSIAN
     - 3.0x
     - 10x
   * - GAUSSIAN_ELLIPSE
     - 2.7x
     - 15x
   * - MULTI_GAUSSIAN (5 components)
     - 8.0x
     - 18x
   * - MULTI_GAUSSIAN_ELLIPSE (5 components)
     - 8.0x
     - 20x
   * - SERSIC
     - 3.4x
     - 10x
   * - SERSIC_ELLIPSE
     - 3.5x
     - 10x
   * - SERSIC_ELLIPSE_Q_PHI
     - 4.0x
     - 10x
   * - SHAPELETS (n_max=6)
     - 8.7x
     - 3.6x
   * - SHAPELETS (n_max=10)
     - 10.0x
     - 4.0x

**Image Convolution**

There is no gaussian convolution function in the JAX library. Thus, in JAXtronomy we construct a gaussian pixel kernel, pad the image, and perform an fft convolution which mimics scipy.ndimage.gaussian_filter with mode="nearest".

When the kernel radius is less than 10, JAXtronomy takes about 1.1x to 1.3x longer than lenstronomy to perform a gaussian convolution, and when the kernel size is larger, it takes JAXtronomy 2x as long or more.
For pixel kernel fft convolution, performance varies significantly with kernel size and grid size.
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





