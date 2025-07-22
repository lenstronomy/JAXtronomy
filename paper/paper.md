---
title: 'JAXtronomy: A JAX port of lenstronomy'
tags:
  - Python
  - astronomy
  - gravitational lensing
  - image simulations
  - dynamics
authors:
  - name: Alan Huang
    orcid: 0009-0001-8629-8826
    affiliation: 1
  - name: Simon Birrer
    orcid: 0000-0003-3195-5507
    affiliation: 1
  - name: Natalie B. Hogg
    orcid: 0000-0001-9346-4477
    affiliation: 2
  - name: Aymeric Galan
    orcid: 0000-0003-2547-9815
    affiliation: 3, 4
  - name: Daniel Gilman
    orcid: 0000-0002-5116-7287
    affiliation: 5
  - name: Anowar J. Shajib
    orcid: 0000-0002-5558-888X
    affiliation: 5, 6, 7
affiliations:
 - name: Department of Physics and Astronomy, Stony Brook University, Stony Brook, NY 1794, USA
   index: 1
 - name: Laboratoire Univers et Particules de Montpellier, CNRS and Université de Montpellier (UMR-5299), 34095 Montpellier, France
   index: 2
 - name: Max-Planck-Institut für Astrophysik, Karl-Schwarzschild Straße 1, 85748 Garching, Germany
   index: 3
 - name: Technical University of Munich, TUM School of Natural Sciences, Physics Department, James-Franck-Straße 1, 85748 Garching, Germany
   index: 4
 - name: Department of Astronomy and Astrophysics, University of Chicago, Chicago, Illinois 60637, USA
   index: 5
 - name: Kavli Institute for Cosmological Physics, University of Chicago, Chicago, IL 60637, USA
   index: 6
 - name: Center for Astronomy, Space Science and Astrophysics, Independent University, Bangladesh, Dhaka 1229, Bangladesh
   index: 7

date:
codeRepository: https://github.com/lenstronomy/JAXtronomy
license: BSD 3-Clause License
bibliography: paper.bib
---

# Summary

`JAXtronomy` is a re-implementation of the gravitational lensing software package `lenstronomy`[^1] [@Birrer:2018; @Birrer:2021] using `JAX`[^2], a Python library that uses an accelerated linear algebra (XLA) compiler to improve the performance of computing software. Our core design principle of `JAXtronomy` is to maintain an identical API to that of `lenstronomy`.

The main `JAX` features utilized in `JAXtronomy` are just-in-time-compilation, which can lead to significant reductions in execution time, and automatic differentiation, which allows for the implementation of gradient-based algorithms that were previously impossible. Additionally, `JAX` allows code to be run on GPUs, further boosting the performance of `JAXtronomy`.

[^1]: https://github.com/lenstronomy/lenstronomy
[^2]: https://github.com/jax-ml/jax

# Statement of need

`lenstronomy` has been widely applied to numerous science cases, with more than 200 publications making use of the software, and has an increasing number of dependent packages relying on features of `lenstronomy`. For instance, science cases directly involving `lenstronomy` include galaxy evolution studies using strong lensing [@Shajib:2021; @DINOS1; @DINOS2] and detailed lens modeling for measuring the Hubble constant using time-delay cosmography by the TDCOSMO collaboration [@TDCOSMO1; @TDCOSMO3; @TDCOSMO4; @TDCOSMO5; @TDCOSMO9; @TDCOSMO18; @TDCOSMO20; @TDCOSMO2025].

Examples of packages dependent on `lenstronomy` for general-purpose lensing computations and image modelling include the `dolphin` package [@Shajib:2025] for automated lens modeling, the `galight` package [@Ding:2020] for galaxy morphology measurements, `SLSim` (Khadka et al, 2025, in prep) for simulating large populations of strong lenses, `pyHalo` [@Gilman:2020] and `mejiro` [@Wedig:2025] for simulating strong lenses with dark matter substructure, and `PALTAS` [@Wagner-Carena:2023] for neural network inference tasks.

In many of these applications, computational constraints are the key limiting factor for strong gravitational lensing science. For example, increased data quality and number of lenses to analyze makes lens modeling a computational bottleneck, and expensive ray-tracing through tens of thousands of dark matter substructures limit the amount of images that can be simulated, especially for the training of neural networks and simulation-based inferences.

These ever-increasing computational costs have lead to the development of several JAX-accelerated strong-lensing packages, such as `gigalens` [@Gu:2022], `herculens` [@Galan:2022], `paltax` [@Wagner-Carena:2024], `GLaD` [@Wang:2025], and Google Research's `jaxstronomy`[^3]. Such packages have been directly inspired by `lenstronomy` and/or support specific use cases. With `JAXtronomy`, we aim to support a wide range of features offered by `lenstronomy` while maintaining an identical API so that packages dependent on `lenstronomy` can transition seamlessly to `JAXtronomy`.

[^3]: https://github.com/google-research/google-research/tree/master/jaxstronomy

# Improvements over lenstronomy in image simulation

The simulation of a lensed image comes in three main steps. The first step begins with a coordinate grid in the angles seen by the observer. These coordinates are ray-traced through the deflectors back to the source plane. This process requires the calculation of light ray deflection angles at each deflector. Second, the surface brightness of the source is calculated on the ray-traced coordinate grid. This produces a lensed image. Third, the lensed image gets convolved by the point spread function (PSF) originating from diffraction of the telescope optics and atmospheric turbulence. Due to the various choices in deflector mass profiles, light model profiles, grid size, and PSF kernel size, the overall runtime of the pipeline can vary significantly.

In the following sections, we outline the improvements in performance that `JAXtronomy` has over `lenstronomy` for each step in the pipeline. These performance benchmarks were run using an Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz, an NVIDIA A100 GPU, and JAX version 0.6.2.

## Deflection angle calculations

Each entry in the table indicates how much faster `JAXtronomy` is compared to `lenstronomy` at computing deflection angles for the corresponding deflector profile and grid size. Those profiles which are already computationally inexpensive for `lenstronomy` are excluded from this table. Some comparisons vary significantly with values of function arguments, so a range is given rather than a number.

| Deflector Profile        | 60x60 grid (cpu) | 180x180 grid (cpu) | 180x180 grid (gpu) |
| :----------------------: | :--------------: | :----------------: | :----------------: |
| CSE                      | 1.6x             | 3.4x               | 3.1x               |
| EPL                      | 5.1x - 15x       | 9.2x - 17x         | 37x - 120x         |
| EPL (jax) vs EPL_NUMBA   | 1.4x             | 3.0x               | 13x                |
| EPL_MULTIPOLE_M1M3M4     | 2.1x - 7x        | 6.8x - 13x         | 42x - 108x         |
| HERNQUIST                | 2.0x             | 3.4x               | 6.4x               |
| HERNQUIST_ELLIPSE_CSE    | 3.8x             | 5.4x               | 40x                |
| MULTIPOLE                | 0.9x             | 1.0x               | 8.3x - 14x         |
| MULTIPOLE_ELL            | 1.5x - 2.1x      | 2.0x - 2.8x        | 70x                |
| NFW                      | 1.6x             | 3.3x               | 4.5x               |
| NFW_ELLIPSE_CSE          | 4.1x             | 6.7x               | 37x                |
| TNFW                     | 2.4x             | 5.8x               | 7.5x               |

## Flux calculations

An analogous table for the different light profiles is shown below.

| Light Profile            | 60x60 grid (cpu) | 180x180 grid (cpu) | 180x180 grid (gpu) |
| :----------------------: | :--------------: | :----------------: | :----------------: |
| CORE_SERSIC              | 2.0x             | 6.7x               | 4.4x               |
| GAUSSIAN                 | 1.0x             | 2.6x               | 1.6x               |
| GAUSSIAN_ELLIPSE         | 1.5x             | 3.7x               | 2.0x               |
| SERSIC                   | 1.0x             | 1.7x               | 4.9x               |
| SERSIC_ELLIPSE           | 1.9x             | 5.8x               | 3.2x               |
| SHAPELETS (n_max=6)      | 6.2x             | 3.4x               | 18x                |
| SHAPELETS (n_max=10)     | 6.0x             | 4.6x               | 22x                |

## FFT Convolution

We find that FFT convolution using `JAX` on CPU results in variable performance boosts or slowdowns compared to `lenstronomy` (which uses `scipy`'s FFT convolution). On a 60x60 grid, and kernel sizes ranging from 3 to 45, `JAX` on CPU ranges from being 1.1x to 2.9x faster than `lenstronomy`, with no obvious correlation to kernel size. On a 180x180 grid, and kernel sizes ranging from 9 to 135, `JAXtronomy` on CPU ranges from being 0.7x to 2.5x as fast as `lenstronomy`, with no obvious correlation to kernel size.

However, FFT convolution using JAX on GPU is significantly faster than `scipy`. On a 60x60 grid, and kernel sizes ranging from 3 to 45, JAX on GPU ranges from being 1.5x to 3.5x faster than `lenstronomy`, with JAX performing better at higher kernel sizes. On a 180x180 grid, and kernel sizes ranging from 9 to 135, `JAXtronomy` on GPU is about 10x to 20x as fast as `lenstronomy`, again with JAX performing better at higher kernel sizes.

# Improvements over lenstronomy in lens modelling

The process of lens modelling involves finding best-fit parameters describing a lensed system from real data. In `lenstronomy`, this typically involves a Particle Swarm Optimizer (PSO) [@Kennedy:1995] for optimization and Monte Carlo Markov Chains for posterior sampling.

`JAXtronomy` retains all of the lens modelling algorithms from `lenstronomy` while benefitting from the increased performance outlined above. Additionally, using JAX's autodifferentiation, we have implemented the L-BFGS gradient descent algorithm from the `Optax`[^4] library [@DeepMind:2020] for optimization. This is a significant improvement over `lenstronomy`'s PSO, which does not have access to gradient information.

[^4]: https://github.com/google-deepmind/optax

# References