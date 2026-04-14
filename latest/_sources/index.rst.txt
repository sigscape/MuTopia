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

Five minutes to MuTopia
-----------------------

The fastest way to get started is to:

1. Pull the docker.
2. Download a pre-trained model from our `Zenodo repository. <https://zenodo.org/records/18803136>`_
3. Apply it to your mutation data. The `annotate-vcf` command infers which topographical mutational processes are active in your sample and annotates each mutation with its most likely generating process.

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

   topo-model setup ${MODEL} ${DATA} ${TUMOR_TYPE}.setup.nc -@ 4
   
   mutopia-sbs annotate-vcf ${MODEL} ${TUMOR_TYPE}.setup.nc ${VCF} --no-pass-only --no-clsuter -fa ${FASTA} -w VAF -o annotated.vcf

MuTopia can do a lot more than just data annotation. 
Check out the tutorials for walkthroughs on data munging, 
model training, and mutational topography analysis!

.. toctree::
   :maxdepth: 2
   :caption: Contents
   :hidden:

   getting_started
   tutorials/index
   api/index
