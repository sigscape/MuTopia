
Getting started with MuTopia
============================

Installation
------------

MuTopia requires Python 3.11 and has a pinned dependency on scikit-learn 1.4.2
because it uses some internal GBT APIs for fast gradient-boosted tree training.
We recommend `uv <https://docs.astral.sh/uv/>`_ to manage the environment — it
resolves and installs dependencies significantly faster than pip alone, and its
lockfile-based workflow makes it easy to reproduce exact environments across machines.

**With Docker (zero setup)**

The fastest way to try MuTopia is with the pre-built Docker image, which ships
with MuTopia and all the bioinformatics tools it depends on (``bedtools``,
``bcftools``, ``tabix``, UCSC ``bigWigAverageOverBed``):

.. code-block:: bash

   docker pull allenlynch/mutopia:latest

   # Mount your data directory and run any CLI command
   docker run --rm -v "$PWD":/workspace allenlynch/mutopia:latest \
       gtensor --help

For interactive use, drop into a shell inside the container:

.. code-block:: bash

   docker run --rm -it -v "$PWD":/workspace allenlynch/mutopia:latest bash

**With uv (recommended for native installs)**

If you don't have uv yet, install it with the official one-liner:

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh

Then create a Python 3.11 virtual environment and install MuTopia from PyPI:

.. code-block:: bash

   uv venv --python 3.11 .venv
   source .venv/bin/activate
   uv pip install mutopia

**With conda**

If you prefer conda, create a fresh environment first to avoid conflicts with
the pinned scikit-learn version:

.. code-block:: bash

   conda create -n mutopia -c conda-forge -y python=3.11
   conda activate mutopia
   pip install mutopia

Verifying the installation
--------------------------

Check that the three command-line tools are available:

.. code-block:: bash

   gtensor --help
   topo-model --help
   mutopia --help

If any of these fail, make sure the virtual environment is active and that its
``bin/`` directory is on your ``PATH``.


Data
----

1. **Genomic features** — Collect feature tracks in BED, bedGraph, or bigWig
   format. MuTopia can ingest any combination of these; see the G-Tensor tutorial
   for details.
2. **Reference genome annotations** — MuTopia needs a FASTA file, a chromsizes
   file, and a blacklist. For hg38 these are included in the tutorial data bundle.
3. **Mutation data** — MuTopia accepts VCF and BCF files. Files split one sample
   per file work best, though multi-sample VCFs are supported via ``-name``.


Basic workflow
--------------

1. **Build G-Tensors** from genomic features and mutation VCFs using ``gtensor compose``.
2. **Train topographic models** on the G-Tensor using ``topo-model train``.
3. **Analyze trained models** interactively with the ``mutopia.analysis`` Python API.
4. **Annotate new samples** by running ``mutopia sbs annotate-vcf`` on any VCF.
