from setuptools import setup, find_packages

tests_require = [
    'pytest',
    'pytest-tornasync',
    'mypy'
]
setup(
    name='cifar10od',
    version='0.1.0',
    author_email='cc@seldon.io',
    license='https://github.com/seldonio/odcd/LICENSE',
    url='https://github.com/seldonio/odcd/integrations/samples/cifar10',
    description='Cifar10 ODCD Server for use inside kfserving ',
    python_requires='>3.4',
    packages=find_packages("cifar10od"),
    install_requires=[
        "odcd",
        "kfserving>=0.2.0",
        "argparse >= 1.4.0",
        "numpy >= 1.8.2",
    ],
    tests_require=tests_require,
    extras_require={'test': tests_require}
)

