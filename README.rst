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

**LensModel ray-tracing**

The table below shows the approximate ratios of jaxtronomy runtimes to lenstronomy runtimes in percentages for different deflector profiles and different grid sizes.

.. list-table::
   :header-rows: 1

   * - Deflector Profile
     - 60x60 grid
     - 180x180 grid
   * - CONVERGENCE
     - 64%
     - 58%
   * - CSE
     - 24%
     - 18%
   * - EPL
     - 8%
     - 8%
   * - EPL (jax) vs EPL_NUMBA
     - 79%
     - 73%
   * - EPL_Q_PHI
     - 322%
     - 321%
   * - GAUSSIAN
     - 47%
     - 46%
   * - GAUSSIAN_POTENTIAL
     - 53%
     - 49%
   * - HERNQUIST
     - 49%
     - 68%
   * - HERNQUIST_ELLIPSE_CSE
     - 24%
     - 25%
   * - LOS
     - 42%
     - 38%
   * - LOS_MINIMAL
     - 39%
     - 39%
   * - NFW
     - 49%
     - 47%
   * - NFW_ELLIPSE_CSE
     - 19%
     - 21%
   * - NIE
     - 77%
     - 81%
   * - PJAFFE
     - 42%
     - 42%
   * - PJAFFE_ELLIPSE_POTENTIAL
     - 32%
     - 32%
   * - SHEAR
     - 54%
     - 49%
   * - SIE
     - 83%
     - 84%
   * - SIS
     - 30%
     - 25%
   * - SPP
     - 69%
     - 65%

**LightModel surface brightness**

The table below shows the approximate ratios of jaxtronomy runtimes to lenstronomy runtimes in percentages for different source profiles and different grid sizes.

.. list-table::
   :header-rows: 1

   * - Source Profile
     - 60x60 grid
     - 180x180 grid
   * - CORE_SERSIC
     - 24%
     - 20%
   * - GAUSSIAN
     - 28%
     - 30%
   * - GAUSSIAN_ELLIPSE
     - 43%
     - 44%
   * - MULTI_GAUSSIAN (5 components)
     - 17%
     - 15%
   * - MULTI_GAUSSIAN_ELLIPSE (5 components)
     - 13%
     - 17%
   * - SERSIC
     - 27%
     - 26%
   * - SERSIC_ELLIPSE
     - 27%
     - 25%
   * - SERSIC_ELLIPSE_Q_PHI
     - 31%
     - 26%
   * - SHAPELETS (n_max = 6)
     - 11%
     - 12%
   * - SHAPELETS (n_max = 10)
     - 5%
     - 5%

Related software packages
-------------------------

The following lensing software packages do use JAX-accelerated computing that in part were inspired or made use of lenstronomy functions:

- Herculens_
- GIGA-lens_
- PaltaX_

.. _Herculens: https://github.com/herculens/herculens
.. _GIGA-lens: https://github.com/giga-lens/gigalens
.. _PaltaX: https://github.com/swagnercarena/paltax





