from setuptools import setup, find_packages

tests_require = [
    'pytest',
    'pytest-tornasync',
    'mypy'
]
setup(
    name='ceserver',
    version='0.1.0',
    author_email='cc@seldon.io',
    license='https://github.com/seldonio/odcd/LICENSE',
    url='https://github.com/seldonio/odcd/integrations/cloudevents',
    description='ODCD cloudevents server',
    python_requires='>3.4',
    packages=find_packages("odcdserver"),
    install_requires=[
        "kfserving>=0.1.0",
        "argparse >= 1.4.0",
        "numpy >= 1.8.2",
        "cloudevents",
    ],
    tests_require=tests_require,
    extras_require={'test': tests_require}
)

