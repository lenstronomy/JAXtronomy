from setuptools import setup, find_packages

requires = ["numpy >= 1.18.0", "lenstronomy >= 1.10.4", "jax", "jaxlib"]
tests_require = ["pytest"]

setup(
    name="JAX-lenstronomy",
    version="0.0.1rc1",
    url="https://github.com/lenstronomy/JAX-lenstronomy",
    author="lenstronomy developers",
    description="lenstronomy, but in JAX",
    packages=find_packages(),
    license="BSD-3",
    install_requires=requires,
    tests_require=tests_require,
    keywords="lenstronomy",
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
