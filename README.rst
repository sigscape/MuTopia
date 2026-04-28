MuTopia — Mutational Topography Inference and Analysis
======================================================

**MuTopia** learns *topographic models* of somatic mutation: it simultaneously
decomposes a cohort's mutation counts into distinct processes (signatures)
**and** explains how local genomic context shapes each signature's activity across
the genome.

Documentation
-------------

Full documentation, API reference, and tutorials are at
`sigscape.github.io/MuTopia <https://sigscape.github.io/MuTopia>`_.

The site includes step-by-step tutorials covering every part of the package:

1. Building a G-Tensor from genomic feature tracks and VCFs
2. Analyzing G-Tensors (slicing, feature management, region queries)
3. Training topographic models (single-fit and Optuna hyperparameter studies)
4. Analyzing trained models (component plots, SHAP, marginal predictions)
5. Genome-browser plotting (composable track views over any region)

System requirements
-------------------

**Software dependencies**

- Python 3.11 (pinned for ``scikit-learn==1.4.2``)
- See ``setup.cfg`` for the complete pinned dependency list.
- CLI bioinformatics tools (auto-installed via Docker / bioconda):
  ``bedtools``, ``bcftools``, ``tabix``, ``samtools``,
  UCSC ``bigWigAverageOverBed``.

**Tested on**

- macOS
- Linux (x86_64)

**Hardware**

MuTopia runs on commodity CPU hardware — no GPU required.
Training a 15-component model on a cohort of ~200 WGS samples uses
~8 GB RAM with default settings; inference and annotation use <4 GB.

Installation
------------

MuTopia requires **Python 3.11** due to a pinned scikit-learn dependency (1.4.2)
used for fast gradient-boosted tree training. We recommend
`uv <https://docs.astral.sh/uv/>`_ — it resolves and installs the full dependency
set in seconds and keeps environments reproducible across machines.

**With Docker (zero setup)** — ~2 minutes:

.. code-block:: bash

   docker pull allenlynch/mutopia:latest
   docker run --rm -v "$PWD":/workspace allenlynch/mutopia:latest gtensor --help

**With uv (recommended for native installs)** — under 30 seconds:

.. code-block:: bash

   # Install uv if you don't have it
   curl -LsSf https://astral.sh/uv/install.sh | sh

   uv venv --python 3.11 .venv
   source .venv/bin/activate
   uv pip install mutopia

**With conda / bioconda** — 2–4 minutes:

MuTopia is published on `bioconda <https://bioconda.github.io/recipes/mutopia/README.html>`_,
which pulls in the bioinformatics tool dependencies (``bedtools``,
``bcftools``, ``tabix``, ``samtools``) automatically:

.. code-block:: bash

   conda create -n mutopia -c conda-forge -c bioconda -y python=3.11 mutopia
   conda activate mutopia

Verify the CLI tools are on your ``PATH``:

.. code-block:: bash

   gtensor --help
   topo-model --help
   mutopia --help


Demo
----

The fastest way to see MuTopia in action is to apply a pre-trained model to a
sample VCF. The ``annotate-vcf`` command infers which topographical mutational
processes are active in your sample and annotates each mutation with its most
likely generating process.

*Note: this is just an example VCF; the results aren't biologically meaningful.*

.. code-block:: bash

   docker pull allenlynch/mutopia:latest

   TUMOR_TYPE="Liver-HCC"
   FASTA="path/to/hg38.fasta"

   ZENODO="https://zenodo.org/records/18803136/files"
   MODEL=${TUMOR_TYPE}.model.pkl
   DATA=${TUMOR_TYPE}.nc
   wget ${ZENODO}/${MODEL}
   wget ${ZENODO}/${DATA}
   wget ${ZENODO}/${DATA}.regions.bed

   VCF=CHC197.sample.hg38.vcf.gz
   wget -O ${VCF} https://github.com/sigscape/MuTopia/releases/download/v1.0.5/CHC197.sample.hg38.vcf.gz

   docker run --rm -v "$PWD":/workspace allenlynch/mutopia:latest \
      topo-model setup ${MODEL} ${DATA} ${TUMOR_TYPE}.setup.nc -@ 4

   docker run --rm -v "$PWD":/workspace -v "$(dirname ${FASTA})":/fasta allenlynch/mutopia:latest \
      mutopia-sbs annotate-vcf ${MODEL} ${TUMOR_TYPE}.setup.nc ${VCF} --no-pass-only --no-cluster -fa /fasta/$(basename ${FASTA}) -w VAF -o annotated.vcf

**Expected output:** ``annotated.vcf`` is a copy of the input VCF with new
``INFO`` fields per record giving the most likely component (signature) for
that mutation and its posterior probability. The full set of annotation fields
is described in the
`annotate-vcf reference <https://sigscape.github.io/MuTopia/stable/cli/mutopia-sbs.html>`_.

**Expected run time:** ~2–3 minutes end-to-end (annotation itself ~30 seconds;
the rest is the one-time G-Tensor download).

Instructions for use
--------------------

To run on your own data:

1. **Annotate a VCF with a pre-trained model** — follow the demo above,
   replacing ``CHC197.sample.hg38.vcf.gz`` with your VCF and choosing the
   tumor-type-matched model from the Zenodo repository.
2. **Train a new model on your cohort** — see Tutorial 3 for the end-to-end
   workflow (build G-Tensor → split → train → score).
3. **Analyze a trained model** — see Tutorial 4 for component plots, SHAP
   feature attribution, and marginal predictions, and Tutorial 5 for
   genome-browser visualization.

Preprint
--------

`Lynch AW, et al. (2026). Topographical archetypes of somatic mutagenesis
in cancer. <https://doi.org/10.64898/2026.04.18.719374>`_
