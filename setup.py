from setuptools import find_packages, setup


def readme():
    with open('README.md') as f:
        return f.read()


# read version file
exec(open('odcd/version.py').read())


setup(name='odcd',
      author='Seldon Technologies Ltd.',
      author_email='hello@seldon.io',
      version=__version__,
      description='Algorithms for outlier detection and concept drift',
      url='https://github.com/SeldonIO/odcd',
      license='Apache 2.0',
      packages=find_packages(),
      include_package_data=True,
      python_requires='>3.5.1',
      setup_requires=[
          'pytest-runner'
      ],
      install_requires=[
          'numpy',
          'tensorflow',
          'tensorflow_probability',
      ],
      tests_require=[
          'pytest',
          'pytest-cov'
      ],
      test_suite='tests',
      zip_safe=False)
