from setuptools import setup, find_packages

tests_require = [
    'pytest',
    'pytest-tornasync',
    'mypy'
]
setup(
    name='signsad',
    version='0.1.0',
    author_email='cc@seldon.io',
    license='https://github.com/seldonio/alibi-detect/LICENSE',
    url='https://github.com/seldonio/alibi-detect/integrations/samples/kfserving/ad-signs',
    description='Traffic Signs Adversarial Detection Server for use inside kfserving ',
    python_requires='>3.4',
    packages=find_packages("signsad"),
    install_requires=[
        "alibi-detect",
        "kfserving>=0.2.0",
        "argparse >= 1.4.0",
        "numpy >= 1.8.2",
        "scikit-learn",
    ],
    tests_require=tests_require,
    extras_require={'test': tests_require}
)

