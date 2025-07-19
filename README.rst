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

To use JAX with an NVIDIA GPU on Linux, an additional first step is needed. Install JAX with ::

  pip install -U "jax[cuda12]"

For other GPUs or operating systems, installation is more complicated.
See the `JAX installation instructions for GPU <https://github.com/jax-ml/jax?tab=readme-ov-file#installation>`_ for more details.

``JAXtronomy`` can be installed with ::

  pip install jaxtronomy

Note that pip installing jaxtronomy without performing the first step will by default install JAX only with CPU support.
**Example notebook**:
`An example notebook <https://github.com/lenstronomy/JAXtronomy/blob/main/notebooks/modeling_a_simple_Einstein_ring.ipynb>`_ has been made available, which
showcases the features and improvements in JAXtronomy.

Performance comparison between JAXtronomy and lenstronomy
---------------------------------------------------------

We compare the runtimes between JAXtronomy and lenstronomy by timing 1,000 function executions.
While lenstronomy is always run using CPU, JAXtronomy can be run using either CPU or GPU.
These tests were run using an Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz, an NVIDIA A100 GPU, and JAX version 0.6.2.
A performance comparison notebook has been made available for reproducibility.

**LensModel ray-shooting**

The table below shows how much faster JAXtronomy is compared to lenstronomy for different deflector profiles and different grid sizes.
Some comparisons vary significantly with values of function arguments.

.. list-table::
  :header-rows: 1

  * - Deflector Profile
    - 60x60 grid (JAX w/ cpu)
    - 180x180 grid (JAX w/ cpu)
    - 180x180 grid (JAX w/ gpu)
  * - CONVERGENCE
    - 0.4x
    - 1.3x
    - 0.5x
  * - CSE
    - 1.6x
    - 3.4x
    - 3.1x
  * - EPL
    - 5.1x - 15x
    - 9.2x - 18x
    - 37x - 120x
  * - EPL (jax) vs EPL_NUMBA
    - 1.4x
    - 3.0x
    - 13x
  * - EPL_MULTIPOLE_M1M3M4
    - 2.1x - 7x
    - 6.8x - 13x
    - 42x - 108x
  * - EPL_MULTIPOLE_M1M3M4_ELL
    - 1.8x - 3.3x
    - 2.6x - 3.3x
    - 120x - 140x
  * - GAUSSIAN
    - 1.4x
    - 3.0x
    - 2.9x
  * - GAUSSIAN_POTENTIAL
    - 1.2x
    - 2.6x
    - 2.4x
  * - HERNQUIST
    - 2.0x
    - 3.4x
    - 6.4x
  * - HERNQUIST_ELLIPSE_CSE
    - 3.8x
    - 5.4x
    - 40x
  * - MULTIPOLE
    - 0.9x
    - 1.0x
    - 8.3x - 14x
  * - MULTIPOLE_ELL (m=1, m=3, m=4)
    - 1.5x, 1.5x, 2.1x
    - 2.0x, 2.0x, 2.8x
    - 75x, 75x, 70x
  * - NFW
    - 1.6x
    - 3.3x
    - 4.5x
  * - NFW_ELLIPSE_CSE
    - 4.1x
    - 6.7x
    - 37x
  * - NIE
    - 0.5x
    - 0.5x
    - 2.0x
  * - PJAFFE
    - 1.0x
    - 1.2x
    - 2.8x
  * - PJAFFE_ELLIPSE_POTENTIAL
    - 1.4x
    - 1.6x
    - 3.1x
  * - SHEAR
    - 0.7x
    - 2.0x
    - 1.0x
  * - SIE
    - 0.5x
    - 0.5x
    - 2.0x
  * - SIS
    - 1.4x
    - 3.3x
    - 2.0x
  * - TNFW
    - 2.4x
    - 5.8x
    - 7.5x

**LightModel surface brightness**

The table below shows how much faster JAXtronomy is compared to lenstronomy for different source profiles and different grid sizes.

.. list-table::
   :header-rows: 1

   * - Source Profile
     - 60x60 grid (JAX w/ cpu)
     - 180x180 grid (JAX w/ cpu)
     - 180x180 grid (JAX w/ gpu)
   * - CORE_SERSIC
     - 2.0x
     - 6.7x
     - 4.4x
   * - GAUSSIAN
     - 1.0x
     - 2.6x
     - 1.6x
   * - GAUSSIAN_ELLIPSE
     - 1.5x
     - 3.7x
     - 2.0x
   * - MULTI_GAUSSIAN (5 components)
     - 3.7x
     - 11x
     - 7.8x
   * - MULTI_GAUSSIAN_ELLIPSE (5 components)
     - 4.0x
     - 14x
     - 8.2x
   * - SERSIC
     - 1.0x
     - 1.7x
     - 4.9x
   * - SERSIC_ELLIPSE
     - 1.9x
     - 5.8x
     - 3.2x
   * - SERSIC_ELLIPSE_Q_PHI
     - 1.7x
     - 5.5x
     - 3.3x
   * - SHAPELETS (n_max=6)
     - 6.2x
     - 3.4x
     - 18x
   * - SHAPELETS (n_max=10)
     - 6.0x
     - 4.6x
     - 23x

**FFT Pixel Kernel Convolution**

Convolution runtimes vary significantly, depending on both grid size and kernel size.

- For a 60x60 grid, and kernel sizes ranging from 3 to 45, jaxtronomy on CPU is about 1.1x to 2.9x faster than lenstronomy, with no obvious correlation to kernel size.
- For a 60x60 grid, and kernel sizes ranging from 3 to 45, jaxtronomy on GPU is about 1.5x to 3.5x faster than lenstronomy, with JAX performing better with higher kernel sizes.
- For a 180x180 grid, and kernel sizes ranging from 9 to 135, jaxtronomy on CPU is about 0.7x to 2.5x as fast as lenstronomy, with no obvious correlation to kernel size.
- For a 180x180 grid, and kernel sizes ranging from 9 t0 135, jaxtronomy on GPU is about 10x to 20x as fast as lenstronomy, with JAX performing better with higher kernel sizes.


Related software packages
-------------------------

The following lensing software packages do use JAX-accelerated computing that in part were inspired or made use of lenstronomy functions:

- Herculens_
- GIGA-lens_
- PaltaX_

.. _Herculens: https://github.com/herculens/herculens
.. _GIGA-lens: https://github.com/giga-lens/gigalens
.. _PaltaX: https://github.com/swagnercarena/paltax


Community guidelines
--------------------

**Contributing to jaxtronomy**

The contributing guidelines are the same as lenstronomy's.
For more details, click `here<https://github.com/lenstronomy/lenstronomy/blob/main/CONTRIBUTING.rst>`_.
In short,

- Fork the repository
- Write clean, well-documented code, following conventions
- Submit pull requests

**Reporting issues, seeking support, and feature requests**

- Submit a Github issue





