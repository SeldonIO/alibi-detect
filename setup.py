from setuptools import find_packages, setup


def readme():
    with open("README.md", encoding="utf-8") as f:
        return f.read()


# read version file
exec(open("alibi_detect/version.py").read())

extras_require = {
    "prophet": [
        "prophet>=1.1.0, <2.0.0",
    ],
    "torch": [
        "torch>=1.7.0, <1.14.0"
    ],
    # https://github.com/SeldonIO/alibi-detect/issues/375 and 387
    "tensorflow": [
        "tensorflow_probability>=0.8.0, <0.23.0",
        "tensorflow>=2.2.0, !=2.6.0, !=2.6.1, <2.15.0",  # https://github.com/SeldonIO/alibi-detect/issues/375 and 387
    ],
    "keops": [
        "pykeops>=2.0.0, <2.2.0",
        "torch>=1.7.0, <1.14.0"
    ],
    "all": [
        "prophet>=1.1.0, <2.0.0",
        "tensorflow_probability>=0.8.0, <0.23.0",
        "tensorflow>=2.2.0, !=2.6.0, !=2.6.1, <2.15.0",  # https://github.com/SeldonIO/alibi-detect/issues/375 and 387
        "pykeops>=2.0.0, <2.2.0",
        "torch>=1.7.0, <1.14.0"
    ],
}

setup(
    name="alibi-detect",
    author="Seldon Technologies Ltd.",
    author_email="hello@seldon.io",
    version=__version__,  # type: ignore # noqa F821
    description="Algorithms for outlier detection, concept drift and metrics.",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/SeldonIO/alibi-detect",
    license="Apache 2.0",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    # lower bounds based on Debian Stable versions where available
    install_requires=[
        "matplotlib>=3.0.0, <4.0.0",
        "numpy>=1.16.2, <2.0.0",
        "pandas>=1.0.0, <3.0.0",
        "Pillow>=5.4.1, <11.0.0",
        "opencv-python>=3.2.0, <5.0.0",
        "scipy>=1.3.0, <2.0.0",
        'scikit-image>=0.19, <0.23',
        "scikit-learn>=0.20.2, <2.0.0",
        "transformers>=4.0.0, <5.0.0",
        "dill>=0.3.0, <0.4.0",
        "tqdm>=4.28.1, <5.0.0",
        "requests>=2.21.0, <3.0.0",
        "pydantic>=1.8.0, <2.0.0",
        "toml>=0.10.1, <1.0.0",  # STC, see https://discuss.python.org/t/adopting-recommending-a-toml-parser/4068
        "catalogue>=2.0.0, <3.0.0",
        "numba>=0.50.0, !=0.54.0, <0.59.0",  # Avoid 0.54 due to: https://github.com/SeldonIO/alibi/issues/466
        "typing-extensions>=3.7.4.3"
    ],
    extras_require=extras_require,
    test_suite="tests",
    zip_safe=False,
    classifiers=[
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering",
    ],
)
