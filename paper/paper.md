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
  - name: Natalie Hogg
    orcid: 0000-0001-9346-4477
    affiliation: 2
  - name: Aymeric Galan
    orcid: 0000-0003-2547-9815
    affiliation: 3, 4
  - name: Daniel Gilman
    orcid: 0000-0002-5116-7287
    affiliation: 5
  - name: Anowar Shajib
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

JAXtronomy is a re-implementation of the gravitational lensing software package lenstronomy [@Birrer:2018; @Birrer:2021] using JAX, a Python library that uses an accelerated linear algebra (XLA) compiler to improve the performance of computing software. Our core design principle of JAXtronomy is to maintain an identical API to that of lenstronomy.

The main JAX features utilized in JAXtronomy are just-in-time-compilation, which can lead to significant reductions in execution time, and automatic differentiation, which allows for the implementation of gradient-based algorithms that were previously impossible. Additionally, JAX allows code to be run using GPU, further boosting the performance of JAXtronomy.

# Statement of need

lenstronomy has found wide applications with more than 200 publications making use of the software, and has an increasing number of dependent packages relying on features of lenstronomy. Examples include the dolphin package [@Shajib:2025] for automated lens modeling, the galight package [@Ding:2020] for galaxy morphology measurements, detailed lens modeling for measuring the Hubble constant using time-delay cosmography by the TDCOSMO collaboration [@TDCOSMO1; @TDCOSMO2; @TDCOSMO3; @TDCOSMO4; @TDCOSMO5], SLSim (Khadka et al, 2025, in prep) for simulating large populations of strong lenses, pyHalo [@Gilman:2020] for simulating strong lenses with dark matter substructure, and PALTAS [@Wagner-Carena:2023] for neural network inference tasks.

In many of these applications, computational constraints are becoming a key limiting factor of strong gravitational lensing sciences. For example, increased data quality and number of lenses to analyze makes lens modeling a computational bottleneck, and expensive ray-tracing through tens of thousands of dark matter substructures limit the amount of images that can be simulated, especially for the training of neural networks and simulation-based inferences.

These ever-increasing computational costs have lead to the development of several JAX-accelerated strong-lensing packages, such as gigalens [@Gu:2022], herculens [@Galan:2022], paltax [@Wagner-Carena:2024], and jaxstronomy\footnote{https://github.com/google-research/google-research/tree/master/jaxstronomy}. Such packages have been directly inspired by lenstronomy and/or support specific use cases. With JAXtronomy, we aim to support a wide range of features offered by lenstronomy while maintaining an identical API so that packages dependent on lenstronomy can transition seamlessly to JAXtronomy.

# Features and Improvements

## Deflection angle calculations

The table below shows how much faster JAXtronomy is compared to lenstronomy at calculating deflection angles for different deflector profiles and different grid sizes. Deflector profiles which are already computationally inexpensive for lenstronomy are excluded from this table.
Some comparisons vary significantly with values of function arguments.

These tests were run using an Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz, an NVIDIA A100 GPU, and JAX version 0.6.2.

| Deflector Profile        | 60x60 (cpu) | 180x180 (cpu) | 180x180 (gpu) |
| :----------------------: | :---------: | :-----------: | :-----------: |
| CSE                      | 1.6x        | 3.4x          | 3.1x          |
| EPL                      | 5.1x - 15x  | 9.2x - 17x    | 37x - 120x    |
| EPL (jax) vs EPL_NUMBA   | 1.4x        | 3.0x          | 13x           |
| EPL_MULTIPOLE_M1M3M4     | 2.1x - 7x   | 6.8x - 13x    | 42x - 108x    |
| HERNQUIST                | 2.0x        | 3.4x          | 6.4x          |
| HERNQUIST_ELLIPSE_CSE    | 3.8x        | 5.4x          | 40x           |
| MULTIPOLE                | 0.9x        | 1.0x          | 8.3x - 14x    |
| MULTIPOLE_ELL            | 1.5x - 2.1x | 2.0x - 2.8x   | 70x           |
| NFW                      | 1.6x        | 3.3x          | 4.5x          |
| NFW_ELLIPSE_CSE          | 4.1x        | 6.7x          | 37x           |
| TNFW                     | 2.4x        | 5.8x          | 7.5x          |

## Flux calculations

Similarly, the table below shows how much faster JAXtronomy is compared to lenstronomy at calculating fluxes for different light source profiles.

| Light Profile            | 60x60 (cpu) | 180x180 (cpu) | 180x180 (gpu) |
| :----------------------: | :---------: | :-----------: | :-----------: |
| CORE_SERSIC              | 2.0x        | 6.7x          | 4.4x          |
| GAUSSIAN                 | 1.0x        | 2.6x          | 1.6x          |
| GAUSSIAN_ELLIPSE         | 1.5x        | 3.7x          | 2.0x          |
| SERSIC                   | 1.0x        | 1.7x          | 4.9x          |
| SERSIC_ELLIPSE           | 1.9x        | 5.8x          | 3.2x          |
| SHAPELETS (n_max=6)      | 6.2x        | 3.4x          | 18x           |
| SHAPELETS (n_max=10)     | 6.0x        | 4.6x          | 22x           |

## FFT Convolution

We find that FFT convolution using JAX on CPU results in variable performance boosts or slowdowns compared to lenstronomy (which uses scipy's FFT convolution). On a 60x60 grid, and kernel sizes ranging from 3 to 45, JAX on CPU ranges from being 1.1x to 2.9x faster than lenstronomy, with no obvious correlation to kernel size. On a 180x180 grid, and kernel sizes ranging from 9 to 135, JAXtronomy on CPU ranges from being 0.7x to 2.5x as fast as lenstronomy, with no obvious correlation to kernel size.

However, FFT convolution using JAX on GPU is significantly faster than scipy. On a 60x60 grid, and kernel sizes ranging from 3 to 45, JAX on GPU ranges from being 1.5x to 3.5x faster than lenstronomy, with JAX performing better at higher kernel sizes. On a 180x180 grid, and kernel sizes ranging from 9 to 135, JAXtronomy on GPU is about 10x to 20x as fast as lenstronomy, again with JAX performing better at higher kernel sizes.

## Lens Modeling

JAXtronomy retains all of the lens modeling algorithms from lenstronomy while benefitting from the increased performance outlined above. Additionally, using JAX's autodifferentiation, we implement scipy's gradient descent minimization algorithm for finding best-fit parameters describing a lensed system. This is a significant improvement over lenstronomy, which does not have access to gradient information and relies on a Particle Swarm Optimizer for finding best-fit parameters.

Although purely JAX-based gradient descent algorithms exist, such as the one in the JAX library itself and in tensorflow, we find that these algorithms become trapped in local minima significantly more often than scipy, due to differences in the line-search algorithm implementation.

# References