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
   :maxdepth: 1
   :caption: Methods

   methods/mahalanobis.ipynb
   methods/iforest.ipynb
   methods/vae.ipynb
   methods/ae.ipynb
   methods/vaegmm.ipynb
   methods/aegmm.ipynb
   methods/prophet.ipynb
   methods/sr.ipynb
   methods/seq2seq.ipynb
   methods/adversarialvae.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/od_mahalanobis_kddcup
   examples/od_if_kddcup
   examples/od_vae_kddcup
   examples/od_vae_cifar10
   examples/od_ae_cifar10
   examples/od_aegmm_kddcup
   examples/od_prophet_weather
   examples/od_sr_synth
   examples/od_seq2seq_synth
   examples/od_seq2seq_ecg
   examples/ad_advvae_mnist

.. toctree::
   :maxdepth: 1
   :caption: API reference

   API reference <api/modules>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
