MuTopia — Mutational Topography Modeling
=========================================

.. image:: website/source/_static/mutopia_logo.png
   :width: 240px
   :align: center

|

**MuTopia** learns *topographic models* of somatic mutation: it simultaneously
decomposes a cancer cohort's mutation counts into distinct processes (signatures)
**and** explains how local genomic context shapes each signature's activity across
the genome.

Unlike standard signature tools that produce per-sample counts, MuTopia produces:

- A genome-wide **topographic map** of each mutational process at 10 kb resolution
- **SHAP feature attributions** linking signature activity to chromatin, replication
  timing, sequence composition, and any other genomic track you supply
- **Per-mutation posterior probabilities** annotated directly in VCF files
- Accurate refitting for **targeted panel and exome** sequencing via exposure correction

Features
--------

==========================================  ================================================
Feature                                     Description
==========================================  ================================================
G-Tensors                                   Spatially-indexed data structure integrating
                                            genomic feature tracks with mutation calls
Topographic NMF                             Gradient-boosted local models + iterative NMF
                                            learn signature × context interactions
SHAP interpretability                       Built-in SHAP beeswarm, interaction matrix,
                                            and track-plot visualizations
VCF annotation                              ``mutopia sbs annotate-vcf`` assigns every
                                            mutation a posterior over components
Panel/exome support                         Whitelist BED corrects exposure for
                                            non-whole-genome sequencing
Hyperparameter tuning                       Optuna-based study management via ``topo-model study``
Interactive track plots                     Composable genome-browser figures in matplotlib
==========================================  ================================================

Installation
------------

MuTopia requires **Python 3.11** due to a pinned scikit-learn dependency (1.4.2)
used for fast gradient-boosted tree training. We recommend
`uv <https://docs.astral.sh/uv/>`_ — it resolves and installs the full dependency
set in seconds and keeps environments reproducible across machines.

**With Docker (zero setup)**

The pre-built image ships with MuTopia plus all the bioinformatics tools it
needs (``bedtools``, ``bcftools``, ``tabix``, UCSC ``bigWigAverageOverBed``):

.. code-block:: bash

   docker pull allenlynch/mutopia:latest
   docker run --rm -v "$PWD":/workspace allenlynch/mutopia:latest gtensor --help

**With uv (recommended for native installs)**

.. code-block:: bash

   # Install uv if you don't have it
   curl -LsSf https://astral.sh/uv/install.sh | sh

   uv venv --python 3.11 .venv
   source .venv/bin/activate
   uv pip install "git+https://github.com/AllenWLynch/Mutopia.git@imports"

**With conda**

.. code-block:: bash

   conda create -n mutopia -c conda-forge -y python=3.11
   conda activate mutopia
   pip install "git+https://github.com/AllenWLynch/Mutopia.git@imports"

Verify the CLI tools are on your ``PATH``:

.. code-block:: bash

   gtensor --help
   topo-model --help
   mutopia --help

Quick start
-----------

.. code-block:: bash

   # 1. Build a G-Tensor from genomic features and mutation VCFs
   gtensor compose config.yaml -w 8

   # 2. Split into train / test by chromosome
   gtensor split data.nc chr1

   # 3. Train a topographic model
   topo-model train -ds data.train.nc data.test.nc -k 15 -o model.pkl -@ 8 --lazy

   # 4. Analyze results in Python
   python - << 'EOF'
   import mutopia.analysis as mu

   model = mu.load_model("model.pkl")
   data  = mu.gt.load_dataset("data.nc", with_samples=False)
   data  = model.annot_data(data, threads=8, calc_shap=True)

   mu.pl.plot_signature_panel(data)
   mu.pl.plot_shap_summary(data, scale=40)
   EOF

   # 5. Annotate individual sample VCFs
   topo-model setup model.pkl data.nc data_setup.nc -@ 8
   mutopia sbs annotate-vcf model.pkl data_setup.nc sample.vcf.gz \
       -m mutation-rate.bedgraph.gz \
       -o sample_annotated.vcf

Documentation
-------------

Full documentation, tutorials, and API reference are at
`sigscape.github.io/MuTopia <https://sigscape.github.io/MuTopia>`_.

Citation
--------

If you use MuTopia in your research, please cite:

.. code-block:: text

   Lynch AW, et al. (2025). MuTopia: topographic modeling of mutational processes
   across the cancer genome. [Preprint]
