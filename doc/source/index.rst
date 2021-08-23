.. alibi-detect documentation master file, created by
   sphinx-quickstart on Thu Feb 28 11:04:41 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. mdinclude:: landing.md

.. toctree::
  :maxdepth: 1
  :caption: Overview

  overview/getting_started
  overview/algorithms
  overview/roadmap

.. toctree::
   :maxdepth: 2
   :caption: Outlier Detection

   od/methods
   od/examples

.. toctree::
   :maxdepth: 2
   :caption: Drift Detection

   cd/background
   cd/methods
   cd/examples

.. toctree::
   :maxdepth: 2
   :caption: Adversarial Detection

   ad/methods
   ad/examples

.. toctree::
   :maxdepth: 1
   :caption: Deployment
   :glob:

   examples/alibi_detect_deploy

.. toctree::
   :maxdepth: 1
   :caption: Datasets

   datasets/overview

.. toctree::
   :maxdepth: 1
   :caption: Models

   models/overview

.. toctree::
   :maxdepth: 1
   :caption: API reference

   API reference <api/modules>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
