
MuTopia Tutorials
=================

Welcome to the MuTopia tutorials. These interactive notebooks guide you through
the key features of MuTopia, from building G-Tensors to training and applying
mutational topographic models.

.. _tutorial-data:

Tutorial data
-------------

All six tutorials share a single data bundle hosted on Zenodo. Download and
unpack it once before starting any tutorial:

.. code-block:: bash

   # Tutorial data bundle (≈ 2 GB): reference genome, pre-built G-Tensor,
   # pre-trained model, and supporting files.
   wget -O tutorial_data.tar.gz https://zenodo.org/records/18803136/files/tutorial_data.tar.gz
   tar -xzf tutorial_data.tar.gz

This produces a ``tutorial_data/`` directory that every tutorial references.

The same Zenodo record also hosts standalone **pre-trained models** for several
tumor types (``<tumor_type>.nc``, ``<tumor_type>.nc.regions.bed`` and ``<tumor_type>.model.pkl``). These live at the
record root and can be downloaded individually — see Tutorial 5 for details.

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   1.building_a_gtensor
   2.analyzing_gtensors
   3.training_models
   4.analyzing_models
   5.annotating_vcfs
   6.genome_browser_plotting
