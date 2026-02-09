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
  - name: Nan Zhang
    orcid: 0000-0002-4861-0081
    affiliation: 8

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
 - name: Department of Physics, University of Illinois, 1110 West Green St., Urbana, IL 61801, USA
   index: 8

date: 28 July 2025
codeRepository: https://github.com/lenstronomy/JAXtronomy
license: BSD 3-Clause License
bibliography: paper.bib
---

# Summary

Gravitational lensing is a phenomenon where light bends around massive objects, resulting in distorted images seen by an observer. Studying graviationally lensed objects can give us key insights into cosmology and astrophysics, such as constraints on the expansion rate of the universe and dark matter models.

Thus, we introduce `JAXtronomy`, a re-implementation of the gravitational lensing software package `lenstronomy`[^1] [@Birrer:2018; @Birrer:2021] using`JAX`[^2]. `JAX` is a Python library that uses an accelerated linear algebra (XLA) compiler to improve the performance of computing software. Our core design principle of `JAXtronomy` is to maintain an identical API to that of `lenstronomy`.

The main `JAX` features utilized in `JAXtronomy` are just-in-time-compilation, which can lead to significant reductions in execution time, and automatic differentiation, which allows for the implementation of gradient-based algorithms that were previously impossible. Additionally, `JAX` allows code to be run on GPUs or parallelized across CPU cores, further boosting the performance of `JAXtronomy`.

[^1]: https://github.com/lenstronomy/lenstronomy
[^2]: https://github.com/jax-ml/jax

# Statement of need

`lenstronomy` has been widely applied to numerous science cases, with more than 200 publications making use of the software, and has an increasing number of dependent packages relying on features of `lenstronomy`. For instance, science cases directly involving `lenstronomy` include galaxy evolution studies using strong lensing [@Shajib:2021; @DINOS1; @DINOS2] and detailed lens modeling for measuring the Hubble constant using time-delay cosmography by the TDCOSMO collaboration [@TDCOSMO1; @TDCOSMO3; @TDCOSMO4; @TDCOSMO5; @TDCOSMO9; @TDCOSMO18; @TDCOSMO20; @TDCOSMO2025].

Examples of packages dependent on `lenstronomy` for general-purpose lensing computations and image modelling include the `dolphin` package [@Shajib:2025] for automated lens modeling, the `galight` package [@Ding:2020] for galaxy morphology measurements, `SLSim` (Khadka et al, 2025, in prep) for simulating large populations of strong lenses, `pyHalo` [@Gilman:2020] and `mejiro` [@Wedig:2025] for simulating strong lenses with dark matter substructure, and `PALTAS` [@Wagner-Carena:2023] for neural network inference tasks.

In many of these applications, computational constraints are the key limiting factor for strong gravitational lensing science. For example, increased data quality and number of lenses to analyze makes lens modeling a computational bottleneck, and expensive ray-tracing through tens of thousands of dark matter substructures limit the amount of images that can be simulated, especially for the training of neural networks and simulation-based inferences. These ever-increasing computational costs have lead to the development of several JAX-accelerated and GPU-accelerated strong-lensing packages, such as `gigalens` [@Gu:2022], `herculens` [@Galan:2022], `paltax` [@Wagner-Carena:2024], `GLaD` [@Wang:2025], `caustics`[^3] [@Stone:2024] and Google Research's `jaxstronomy`[^4].

[^3]: https://github.com/Ciela-Institute/caustics
[^4]: https://github.com/google-research/google-research/tree/master/jaxstronomy

## Why JAXtronomy?

`JAXtronomy` inherits a wide range of features from `lenstronomy` that are not offered by any of the aforementioned JAX-accelerated or GPU-accelerated software. Such features include `lenstronomy`'s linear amplitude solver, which reduces the number of sampled parameters during lens modeling, as well as a variety of log likelihood functions and optional punishment terms to improve robustness during fitting. Furthermore, `JAXtronomy` aims to maintain an identical API to `lenstronomy` so that packages dependent on `lenstronomy` can transition seamlessly to `JAXtronomy`.

# Improvements over lenstronomy in image simulation

The simulation of a lensed image comes in three main steps. The first step begins with a coordinate grid in the angles seen by the observer. These coordinates are ray-traced through the deflectors back to the source plane. This process requires the calculation of light ray deflection angles at each deflector. Second, the surface brightness of the source is calculated on the ray-traced coordinate grid. This produces a lensed image. Third, the lensed image gets convolved by the point spread function (PSF) originating from diffraction of the telescope optics and atmospheric turbulence. Due to the various choices in deflector mass profiles, light model profiles, grid size, and PSF kernel size, the overall runtime of the pipeline can vary significantly.

In the following sections, we outline the improvements in performance that `JAXtronomy` has over `lenstronomy` for each step in the pipeline. These performance benchmarks were run using an Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz, an NVIDIA A100 GPU, and `JAX` version 0.7.0.

## Deflection angle calculations

Each entry in the table indicates how much faster `JAXtronomy` is compared to `lenstronomy` at computing deflection angles for the corresponding deflector profile and grid size. Some comparisons vary significantly with values of function arguments, so a range is given rather than a number.

| Deflector Profile        | 60x60 grid (cpu) | 180x180 grid (cpu) | 180x180 grid (gpu) |
| :----------------------: | :--------------: | :----------------: | :----------------: |
| CONVERGENCE              | 0.4x             | 1.1x               | 0.5x               |
| CSE                      | 1.6x             | 2.6x               | 2.6x               |
| EPL                      | 5.1x - 15x       | 9.2x - 17x         | 37x - 120x         |
| EPL (jax) vs EPL_NUMBA   | 1.4x             | 3.0x               | 13x                |
| EPL_MULTIPOLE_M1M3M4     | 2.1x - 7x        | 6.4x - 13x         | 42x - 108x         |
| HERNQUIST                | 2.0x             | 3.4x               | 5.8x               |
| HERNQUIST_ELLIPSE_CSE    | 3.8x             | 5.4x               | 40x                |
| MULTIPOLE                | 0.9x             | 1.0x               | 8.3x - 14x         |
| MULTIPOLE_ELL            | 1.5x - 2.1x      | 2.0x - 2.8x        | 70x                |
| NIE/SIE                  | 0.5x             | 0.5x               | 2.0x               |
| NFW                      | 1.6x             | 3.3x               | 4.5x               |
| NFW_ELLIPSE_CSE          | 4.1x             | 6.7x               | 31x                |
| PJAFFE                   | 1.0x             | 1.2x               | 2.8x               |
| PJAFFE_ELLIPSE_POTENTIAL | 1.4x             | 1.6x               | 3.1x               |
| SHEAR                    | 0.7x             | 2.0x               | 0.9x               |
| SIS                      | 1.4x             | 3.3x               | 2.0x               |
| TNFW                     | 2.4x             | 5.8x               | 7.5x               |

For small enough grid sizes, `JAXtronomy` computes deflection angles slower than `lenstronomy` when using certain deflector profiles. This is because function call overheads are significantly higher in JAX than in standard Python, so computations that are already fast in Python can end up slower in JAX. In these cases, the benefit of using JAX is to have automatic differentiation for lens modeling.

## Flux calculations

An analogous table for the different light profiles is shown below. The MULTI_GAUSSIAN and MULTI_GAUSSIAN_ELLIPSE profiles include five GAUSSIAN and GAUSSIAN_ELLIPSE components, respectively, highlighting JAX's improved performance in sequential computations.

| Light Profile          | 60x60 grid (cpu) | 180x180 grid (cpu) | 180x180 grid (gpu) |
| :--------------------: | :--------------: | :----------------: | :----------------: |
| CORE_SERSIC            | 2.0x             | 6.7x               | 4.2x               |
| GAUSSIAN               | 1.0x             | 2.5x               | 1.3x               |
| GAUSSIAN_ELLIPSE       | 1.5x             | 3.6x               | 2.0x               |
| MULTI_GAUSSIAN         | 3.7x             | 11x                | 7.8x               |
| MULTI_GAUSSIAN_ELLIPSE | 4.0x             | 13x                | 6.9x               |
| SERSIC                 | 1.0x             | 1.7x               | 3.9x               |
| SERSIC_ELLIPSE         | 1.9x             | 5.7x               | 3.2x               |
| SERSIC_ELLIPSE_Q_PHI   | 1.7x             | 5.5x               | 3.3x               |
| SHAPELETS (n_max=6)    | 6.2x             | 3.4x               | 15x                |
| SHAPELETS (n_max=10)   | 6.0x             | 4.5x               | 17x                |

## FFT Convolution

We find that FFT convolution using `JAX` on CPU results in variable performance boosts or slowdowns compared to `lenstronomy` (which uses `scipy`'s FFT convolution). On a 60x60 grid, and kernel sizes ranging from 3 to 45, `JAX` on CPU ranges from being 1.1x to 2.9x faster than `lenstronomy`, with no obvious correlation to kernel size. On a 180x180 grid, and kernel sizes ranging from 9 to 135, `JAXtronomy` on CPU ranges from being 0.7x to 2.5x as fast as `lenstronomy`, with no obvious correlation to kernel size.

However, FFT convolution using `JAX` on GPU is significantly faster than `scipy`. On a 60x60 grid, and kernel sizes ranging from 3 to 45, JAX on GPU ranges from being 1.5x to 3.5x faster than `lenstronomy`, with `JAX` performing better at higher kernel sizes. On a 180x180 grid, and kernel sizes ranging from 9 to 135, `JAXtronomy` on GPU is about 10x to 20x as fast as `lenstronomy`, again with `JAX` performing better at higher kernel sizes.

# Improvements over lenstronomy in lens modelling

The process of lens modelling involves finding best-fit parameters describing a lensed system from real data. In `lenstronomy`, this typically involves a Particle Swarm Optimizer (PSO) [@Kennedy:1995] for optimization and Monte Carlo Markov Chains for posterior sampling. `JAXtronomy` retains these lens modelling algorithms from `lenstronomy` while benefitting from the increased performance outlined above.

In the following table, we compare `JAXtronomy`'s PSO performance to that of `lenstronomy` when modeling a lens with an elliptical power-law (EPL) mass profile, Sersic light profile, and a quadruply-imaged point source. The image is simulated using a 100x100 grid and FFT convolved using a PSF kernel with a size of 13 pixels. These benchmarks were performed using the same hardware as in the previous section.

| Device       | 64 Particles | 128 Particles | 256 Particles | 512 Particles |
| :-------- -: | :----------: | :-----------: | :-----------: | :-----------: |
| 1 CPU core   | 4x           | 4x            | 5x            | 5x            |
| 2 CPU cores  | 6x           | 7x            | 9x            | 8x            |
| 4 CPU cores  | 11x          | 11x           | 17x           | 15x           |
| 8 CPU cores  | 14x          | 17x           | 24x           | 30x           |
| 16 CPU cores | 16x          | 21x           | 33x           | 38x           |
| 32 CPU cores | 16x          | 18x           | 30x           | 34x           |
| GPU          | 5x           | 6x            | 9x            | 11x           |

The following table shows the same comparison but with the EPL mass profile replaced by a singular isothermal ellipsoid (SIE).

| Device       | 64 Particles | 128 Particles | 256 Particles | 512 Particles |
| :-------- -: | :----------: | :-----------: | :-----------: | :-----------: |
| 1 CPU core   | 3x           | 3x            | 3x            | 4x            |
| 2 CPU cores  | 5x           | 6x            | 6x            | 7x            |
| 4 CPU cores  | 8x           | 12x           | 11x           | 12x           |
| 8 CPU cores  | 11x          | 17x           | 17x           | 24x           |
| 16 CPU cores | 13x          | 20x           | 22x           | 29x           |
| 32 CPU cores | 13x          | 20x           | 20x           | 29x           |
| GPU          | 8x           | 7x            | 27x           | 46x           |


Additionally, using `JAX`'s autodifferentiation, we have implemented the L-BFGS gradient descent algorithm from the `Optax`[^5] library [@DeepMind:2020] for optimization. This is a significant improvement over `lenstronomy`'s PSO, which does not have access to gradient information. Due to the random nature of the PSO, we do not present a concrete comparison between `lenstronomy` and `JAXtronomy` for how long it takes to find best-fit parameters. However, we note that `JAXtronomy` can find a good fit within one minute, while `lenstronomy` can take hours.

[^5]: https://github.com/google-deepmind/optax

# References