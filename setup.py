from setuptools import setup, find_packages

requires = [
    "lenstronomy>=1.13.5",
    "numpy>=2.0.0",
    "jax>=0.7.0",
    "numpyro>=0.19.0",
    "optax>=0.2.5",
    "tqdm",
    "scipy",
    "dynesty",
    "zeus-mcmc",
    "cobaya",
    "nautilus-sampler>=0.7",
    "emcee>=3.0.0",
]
tests_require = ["pytest"]
readme = open("README.rst").read()

setup(
    name="jaxtronomy",
    version="0.1.2",
    python_requires=">=3.11",
    url="https://github.com/lenstronomy/JAXtronomy",
    author="jaxtronomy developers",
    description="lenstronomy, but in JAX",
    long_description=readme,
    packages=find_packages(),
    license="BSD-3",
    install_requires=requires,
    tests_require=tests_require,
    keywords="jaxtronomy",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.11",
    ],
)
