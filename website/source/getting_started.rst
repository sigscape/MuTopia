
Getting started with MuTopia
============================

Installation
------------

Install MuTopia from github using pip. First create a new environment - we have a hard requirement on scikit-learn because we used 
some internal APIs to make speed up training with gradient boosting trees.

.. code-block:: bash
    
    $ conda create --name mutopia -c conda-forge -y python=3.11 && conda activate mutopia
    $ pip install git+https://github.com/AllenWLynch/Mutopia.git@imports

Check that the command line tools are installed:

.. code-block:: bash

    $ gtensor --help
    $ topo-model --help


Data
----

1. Genomic features: You can download or collect genomic features in whatever formats you like. MuTopia can ingest bedfiles, bigWig files, and bedGraphfiles. See the tutorial on building G-Tensors for more details.
2. Reference genome annotations: MuTopia needs some reference genome annotations including a fasta file, chromsizes files, and a blacklist. For the human hg38 genome, you can get these essential materials from the tutorial data!
3. Mutation data: MuTopia supports VCF and BCF files. It's easiest if those files are split by sample. 

Basic workflow
--------------

1. Build G-Tensors from genomic features using the `gtensor` command line tool or the `mutopia.gtensor` module.
2. Train topographical mutational models using the `topo-model` command line.
3. Analyze trained models using the interactive python `mutopia.analysis` module.