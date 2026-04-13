MuTopia — Mutational Topography Modeling
=========================================

.. image:: _static/mutopia_logo.png
   :align: center
   :width: 240px

|

**MuTopia** is a Python toolkit for studying how mutational signatures vary across the
genome. It learns *topographic models* that jointly explain which mutational processes
are active in a cancer genome and how local genomic context (chromatin state, replication
timing, transcription level, DNA sequence composition, …) shapes the rate of each process
across the genome.

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
      gradient-boosted local models and iterative NMF refitting. Tune hyperparameters
      automatically with Optuna.

   .. grid-item-card:: Annotate VCFs
      :link: tutorials/5.annotating_vcfs
      :link-type: doc

      Apply a trained model to any VCF — including panel and exome data via the
      whitelist feature — and assign every mutation a posterior probability over
      model components.

Why MuTopia?
------------

Standard mutational signature tools (SigProfiler, SigFit, deconstructSigs, …) assign
signatures to *samples*. MuTopia goes further: it asks **why** each locus in the genome
has the mutation rate that it does, and which genomic features drive that variation.

Key capabilities:

- **Topographic resolution** — Learns how local chromatin, sequence, and regulatory
  features modulate each signature's activity across 10 kb windows genome-wide.
- **SHAP interpretability** — Attributes locus-level rate variation to individual
  features with SHAP values from the underlying gradient-boosted tree models.
- **Per-mutation posteriors** — Assigns every somatic mutation to the most likely
  generating process, supporting downstream variant prioritization and filtering.
- **Panel/exome support** — The whitelist feature corrects for unobserved territory,
  enabling accurate refitting on targeted sequencing data.
- **Interactive visualization** — Composable genome-browser track plots and
  publication-quality signature panels included out of the box.

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

We recommend `uv <https://docs.astral.sh/uv/>`_ for fast, reproducible installs.
MuTopia requires Python 3.11 due to a pinned scikit-learn dependency.

.. code-block:: bash

   uv venv --python 3.11 .venv && source .venv/bin/activate
   uv pip install "git+https://github.com/AllenWLynch/Mutopia.git@imports"

Conda works too — see :doc:`getting_started` for both options. Verify the install:

.. code-block:: bash

   gtensor --help && topo-model --help && mutopia --help

.. toctree::
   :maxdepth: 2
   :caption: Contents
   :hidden:

   getting_started
   tutorials/index
   api/index
