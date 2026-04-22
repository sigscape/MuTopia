MuTopia — Mutational Topography Modeling
=========================================

**MuTopia** learns *topographic models* of somatic mutation: it simultaneously
decomposes a cohort's mutation counts into distinct processes (signatures)
**and** explains how local genomic context shapes each signature's activity across
the genome.

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
   uv pip install mutopia

**With conda / bioconda**

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


Five minutes to MuTopia
-----------------------

The fastest way to get started is to:

1. Pull the docker.
2. Download a pre-trained model from our `Zenodo repository. <https://zenodo.org/records/18803136>`_
3. Apply it to your mutation data. The `annotate-vcf` command infers which topographical mutational processes are active in your sample and annotates each mutation with its most likely generating process.

*Note: this is just an example VCF, so the results aren't meaningful.*

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

MuTopia can do a lot more than just data annotation. 
Check out the tutorials for walkthroughs on data munging, 
model training, and mutational topography analysis!

Documentation
-------------

Full documentation, tutorials, and API reference are at
`sigscape.github.io/MuTopia <https://sigscape.github.io/MuTopia>`_.
