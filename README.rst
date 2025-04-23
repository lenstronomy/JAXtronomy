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

We compare the runtimes between JAXtronomy and lenstronomy by timing 10,000 function executions.
While lenstronomy is always run using CPU, JAXtronomy can be run using either CPU or GPU.

**LensModel ray-shooting**

The table below shows how much faster JAXtronomy is compared to lenstronomy for different deflector profiles and different grid sizes.
These tests were run using an Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz and an NVIDIA A100 GPU.

.. list-table::
  :header-rows: 1

  * - Deflector Profile
    - 60x60 grid (JAX w/ cpu)
    - 180x180 grid (JAX w/ cpu)
    - 180x180 grid (JAX w/ gpu)
  * - CONVERGENCE
    - 0.4x
    - 1.3x
    - 0.4x
  * - CSE
    - 1.6x
    - 2.9x
    - 2.3x
  * - EPL (e1 = e2 = 0.0)
    - 0.1x
    - 0.2x
    - 1.6x
  * - EPL (e1 = e2 = 0.5)
    - 6.7x
    - 10.8x
    - 76.6x
  * - EPL (jax) vs EPL_NUMBA (e1 = e2 = 0.0)
    - 0.2x
    - 0.3x
    - 2.7x
  * - EPL (jax) vs EPL_NUMBA (e1 = e2 = 0.5)
    - 0.8x
    - 1.5x
    - 11.3x
  * - GAUSSIAN
    - 1.0x
    - 1.8x
    - 3.0x
  * - GAUSSIAN_POTENTIAL
    - 0.9x
    - 1.7x
    - 2.4x
  * - HERNQUIST
    - 1.9x
    - 3.6x
    - 6.4x
  * - HERNQUIST_ELLIPSE_CSE
    - 3.8x
    - 5.9x
    - 40.3x
  * - NFW
    - 1.6x
    - 3.3x
    - 5.0x
  * - NFW_ELLIPSE_CSE
    - 4.1x
    - 5.7x
    - 36.5x
  * - NIE
    - 0.5x
    - 0.5x
    - 2.0x
  * - PJAFFE
    - 1.0x
    - 1.2x
    - 3.0x
  * - PJAFFE_ELLIPSE_POTENTIAL
    - 1.5x
    - 1.6x
    - 3.1x
  * - SHEAR
    - 0.7x
    - 2.2x
    - 1.0x
  * - SIE
    - 0.5x
    - 0.5x
    - 2.0x
  * - SIS
    - 1.4x
    - 3.0x
    - 2.0x
  * - SPP
    - 0.5x
    - 1.0x
    - 2.9x
  * - TNFW
    - 2.4x
    - 5.4x
    - 8.3x

Note that some profiles' runtime may be dependent on function arguments. For example, the EPL profile involves performing a hyp2f1 calculation using a power series expansion.
In lenstronomy, the number of terms used depends on how quickly the series converges, whereas in JAXtronomy, the power series always involves a fixed number of terms, which is required for autodifferentiation.

A performance comparison notebook is available for more detailed analysis.

**LightModel surface brightness**

The table below shows how much faster JAXtronomy is compared to lenstronomy for different source profiles and different grid sizes.
These tests were run using an Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz and an NVIDIA A100 GPU.

.. list-table::
   :header-rows: 1

   * - Source Profile
     - 60x60 grid (JAX w/ cpu)
     - 180x180 grid (JAX w/ cpu)
     - 180x180 grid (JAX w/ gpu)
   * - CORE_SERSIC
     - 2.1x
     - 10.2x
     - 4.4x
   * - GAUSSIAN
     - 1.6x
     - 3.4x
     - 1.6x
   * - GAUSSIAN_ELLIPSE
     - 1.5x
     - 6.9x
     - 2.1x
   * - MULTI_GAUSSIAN (5 components)
     - 3.7x
     - 16.2x
     - 7.8x
   * - MULTI_GAUSSIAN_ELLIPSE (5 components)
     - 4.4x
     - 18.3x
     - 7.2x
   * - SERSIC
     - 2.3x
     - 9.3x
     - 4.2x
   * - SERSIC_ELLIPSE
     - 2.1x
     - 8.5x
     - 3.2x
   * - SERSIC_ELLIPSE_Q_PHI
     - 1.7x
     - 8.6x
     - 3.4x
   * - SHAPELETS (n_max=6)
     - 8.0x
     - 5.2x
     - 17.6x
   * - SHAPELETS (n_max=10)
     - 8.9x
     - 6.1x
     - 22.4x

**FFT Pixel Kernel Convolution**

Convolution runtimes vary significantly, depending on both grid size and kernel size. A short summary is as follows, using
an Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz and an NVIDIA A100 GPU.

- For a 60x60 grid, and kernel sizes ranging from 3 to 45, jaxtronomy on CPU is about 1.1x to 2.9x faster than lenstronomy, with no obvious correlation to kernel size.
- For a 60x60 grid, and kernel sizes ranging from 3 to 45, jaxtronomy on GPU is about 1.5x to 3.5x faster than lenstronomy, with JAX performing better with higher kernel sizes.
- For a 180x180 grid, and kernel sizes ranging from 9 to 135, jaxtronomy on CPU is about 0.7x to 2.5x as fast as lenstronomy, with no obvious correlation to kernel size.
- For a 180x180 grid, and kernel sizes ranging from 9 t0 135, jaxtronomy on GPU is about 10x to 20x as fast as lenstronomy, with JAX performing better with higher kernel sizes.

A performance comparison notebook is available for more detailed analysis.

Related software packages
-------------------------

The following lensing software packages do use JAX-accelerated computing that in part were inspired or made use of lenstronomy functions:

- Herculens_
- GIGA-lens_
- PaltaX_

.. _Herculens: https://github.com/herculens/herculens
.. _GIGA-lens: https://github.com/giga-lens/gigalens
.. _PaltaX: https://github.com/swagnercarena/paltax





