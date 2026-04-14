MuTopia — Mutational Topography Inference and Analysis
======================================================

**MuTopia** is a Python toolkit for studying how mutational signatures vary across the
genome. It learns *topographic models* that jointly explain which mutational processes
are active in a genome and how local genomic context (chromatin state, replication
timing, transcription level, DNA sequence composition, …) shapes process's mutation rate
profiles.

.. grid:: 3

   .. grid-item-card:: Build G-Tensors
      :link: tutorials/1.building_a_gtensor
      :link-type: doc

      Integrate any combination of genomic feature tracks (bigWig, BED, bedGraph)
      with mutation calls into a spatially-indexed G-Tensor — the core data structure
      behind every MuTopia analysis.

   .. grid-item-card:: Train Models
      :link: tutorials/3.training_models
      :link-type: doc

      Decompose mutation counts into topographically-resolved components with
      expressive, nonlinear rate models. Tune hyperparameters
      automatically with Optuna.

   .. grid-item-card:: Annotate Data
      :link: tutorials/5.annotating_vcfs
      :link-type: doc

      Apply a trained model to any VCF — including panel and exome data - to learn
      which topographic processes are driving mutagenes. Go a step further and annotate
      each mutation with its most likely generating process. 

Quick example
-------------

.. code-block:: python

   import mutopia.analysis as mu

   # Load a pre-trained model and annotate a G-Tensor
   model = mu.load_model("model.pkl")
   data  = mu.gt.load_dataset("Liver.nc", with_samples=False)
   data  = model.annot_data(data, threads=8, calc_shap=True)

   # Signature panels, SHAP summary, and track plots
   mu.pl.plot_signature_panel(data)
   mu.pl.plot_shap_summary(data, scale=40)

.. code-block:: bash

   # Annotate a VCF with per-mutation component posteriors
   mutopia sbs annotate-vcf model.pkl Liver_setup.nc sample.vcf.gz \
       -m mutation-rate.hg38.bedgraph.gz \
       -o sample_annotated.vcf

Installation
------------

MuTopia requires Python 3.11 due to a pinned scikit-learn dependency.

The easiest install is via `bioconda <https://bioconda.github.io/recipes/mutopia/README.html>`_,
which also pulls in the bioinformatics tool dependencies (``bedtools``,
``bcftools``, ``tabix``, ``samtools``):

.. code-block:: bash

   conda create -n mutopia -c conda-forge -c bioconda -y python=3.11 mutopia
   conda activate mutopia

Or with `uv <https://docs.astral.sh/uv/>`_ for a fast pip-based install:

.. code-block:: bash

   uv venv --python 3.11 .venv && source .venv/bin/activate
   uv pip install mutopia

See :doc:`getting_started` for the full set of options. Verify the install:

.. code-block:: bash

   gtensor --help && topo-model --help && mutopia --help

.. toctree::
   :maxdepth: 2
   :caption: Contents
   :hidden:

   getting_started
   tutorials/index
   api/index
