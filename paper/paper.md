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
  - name: Aymeric Galan
    orcid: 0000-0003-2547-9815
    affiliation: 2, 3
  - name: Natalie Hogg
    orcid: 0000-0001-9346-4477
    affiliation: 4
  - name: Anowar Shajib
    orcid: 0000-0002-5558-888X
    affiliation: 5, 6, 7
affiliations:
 - name: Department of Physics and Astronomy, Stony Brook University, USA
   index: 1
 - name: Max-Planck-Institut für Astrophysik, Germany
   index: 2
 - name: Max-Planck-Institut für Astrophysik, Germany Technical University of Munich, TUM School of Natural Sciences Physics Department, Germany
   index: 3
 - name: Laboratoire univers et particules de Montpellier, Université de Montpellier, France
   index: 4
 - name: Department of Astronomy and Astrophysics, University of Chicago, USA
   index: 5
 - name: Kavli Institute for Cosmological Physics, University of Chicago, USA
   index: 6
 - name: Center for Astronomy, Space Science and Astrophysics, Independent University, Bangladesh
   index: 7

date:
codeRepository: https://github.com/lenstronomy/JAXtronomy
license: BSD 3-Clause License
bibliography: paper.bib
---

# Summary

JAXtronomy is a re-implementation of the lenstronomy software package [@Birrer:2018; @Birrer:2021] using JAX, a Python library that uses an accelerated linear algebra (XLA) compiler to improve the performance of computing software. Our core design principle of JAXtronomy is to maintain identical API to that of lenstronomy.

The main JAX features utilized in JAXtronomy are just-in-time-compilation, which can lead to significant reductions in execution time, and automatic differentiation, which allows for the implementation of gradient-based algorithms that were previously impossible. Additionally, JAX allows code to be run using GPU, further boosting the performance of JAXtronomy.

# Statement of need

lenstronomy has found wide applications with more than 200 publications making use of the software, and has an increasing number of dependent packages relying on features of lenstronomy. Examples include the galight package [@Ding:2020] for lens modeling, the TDCOSMO group for measuring the Hubble constant using time-delay cosmography [@TDCOSMO1; @TDCOSMO2; @TDCOSMO3; @TDCOSMO4; @TDCOSMO5] for simulating large populations of strong lenses, and pyHalo [@Gilman:2020] for simulating strong lenses with dark matter substructure.

In many of these applications, computational constraints are becoming a key limiting factor of strong gravitational lensing sciences. For example, increased data quality and number of lenses to analyze limit the efficiency of lens modeling, and expensive ray-tracing through tens of thousands of dark matter substructures limit the amount of images that can be simulated, especially for the training of neural networks.

These ever-increasing computational costs have lead to the development of several JAX-accelerated strong-lensing packages, such as gigalens [@Gu:2022], herculens [@Galan:2022], paltax [@Wagner-Carena:2021], and jaxstronomy\footnote{https://github.com/google-research/google-research/tree/master/jaxstronomy}. Such packages have been directly inspired by lenstronomy and/or support specific use cases. With JAXtronomy, we aim to support a wide range of features offered by lenstronomy while maintaining an identical API so that packages dependent on lenstronomy can transition seamlessly to JAXtronomy.

# Features and Improvements

## Deflection angle calculations

The table below shows how much faster JAXtronomy is compared to lenstronomy at calculating deflection angles for different deflector profiles and different grid sizes.
Some comparisons vary significantly with values of function arguments.

These tests were run using an Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz, an NVIDIA A100 GPU, and JAX version 0.5.2.

| Deflector Profile        | 60x60 (cpu) | 180x180 (cpu) | 180x180 (gpu) |
| :----------------------: | :---------: | :-----------: | :-----------: |
| Convergence              | 0.4x        | 1.3x          | 0.4x          |
| CSE                      | 1.6x        | 2.9x          | 2.3x          |
| EPL                      | 1.1x - 15x  | 1.6x - 17x    | 6.4x - 120x   |
| EPL (jax) vs EPL_NUMBA   | 1.8x        | 3.2x          | 13x           |
| EPL_MULTIPOLE_M1M3M4     | 1.1x - 7x   | 3.3x - 13x    | 22x - 108x    |
| GAUSSIAN                 | 1.0x        | 1.8x          | 3.0x          |
| GAUSSIAN_POTENTIAL       | 0.9x        | 1.7x          | 2.4x          |
| HERNQUIST                | 1.9x        | 3.6x          | 6.4x          |
| HERNQUIST_ELLIPSE_CSE    | 3.8x        | 5.9x          | 40x           |
| MULTIPOLE                | 0.9x        | 1.0x          | 10.0x         |
| MULTIPOLE_ELL            | 1.5x - 2.1x | 1.3x - 1.9x   | 90x           |
| NFW                      | 1.6x        | 3.3x          | 5.0x          |
| NFW_ELLIPSE_CSE          | 4.1x        | 5.7x          | 36x           |
| NIE (and SIE)            | 0.5x        | 0.5x          | 2.0x          |
| PJAFFE                   | 1.0x        | 1.2x          | 3.0x          |
| PJAFFE_ELLIPSE_POTENTIAL | 1.5x        | 1.6x          | 3.1x          |
| SHEAR                    | 0.7x        | 2.2x          | 1.0x          |
| SIS                      | 1.4x        | 3.0x          | 2.0x          |
| TNFW                     | 2.4x        | 5.4x          | 8.3x          |


## Flux calculations

Similarly, the table below shows how much faster JAXtronomy is compared to lenstronomy at calculating fluxes for different light source profiles.

| Light Profile            | 60x60 (cpu) | 180x180 (cpu) | 180x180 (gpu) |
| :----------------------: | :---------: | :-----------: | :-----------: |
| CORE_SERSIC              | 2.1x        | 10.2x         | 4.4x          |
| GAUSSIAN                 | 1.6x        | 3.4x          | 1.6x          |
| GAUSSIAN_ELLIPSE         | 1.5x        | 6.9x          | 2.1x          |
| SERSIC                   | 2.3x        | 9.3x          | 4.2x          |
| SERSIC_ELLIPSE           | 2.1x        | 8.5x          | 3.2x          |
| SHAPELETS (n_max=6)      | 8.0x        | 5.2x          | 17.6x         |
| SHAPELETS (n_max=10)     | 8.9x        | 6.1x          | 22.4x         |

## FFT Convolution

We find that FFT convolution using JAX on CPU results in variable performance boosts or slowdowns compared to lenstronomy (which uses scipy's FFT convolution). On a 60x60 grid, and kernel sizes ranging from 3 to 45, JAX on CPU ranges from being 1.1x to 2.9x faster than lenstronomy, with no obvious correlation to kernel size. On a 180x180 grid, and kernel sizes ranging from 9 to 135, JAXtronomy on CPU ranges from being 0.7x to 2.5x as fast as lenstronomy, with no obvious correlation to kernel size.

However, FFT convolution using JAX on GPU is significantly faster than scipy. On a 60x60 grid, and kernel sizes ranging from 3 to 45, JAX on GPU ranges from being 1.5x to 3.5x faster than lenstronomy, with JAX performing better at higher kernel sizes. On a 180x180 grid, and kernel sizes ranging from 9 t0 135, JAXtronomy on GPU is about 10x to 20x as fast as lenstronomy, again with JAX performing better at higher kernel sizes.

## Lens Modeling

Using JAX's autodifferentiation, we implement scipy's gradient descent minimization algorithm for finding best-fit parameters describing a lensed system. This is a significant improvement over lenstronomy, which does not have access to gradient information and relies on a Particle Swarm Optimizer for finding best-fit parameters. 

Although purely JAX-based gradient descent algorithms exist, such as the one in the JAX library itself and in tensorflow, we find that these algorithms become trapped in local minima significantly more often than scipy, due to differences in the line-search algorithm implementation.

# References