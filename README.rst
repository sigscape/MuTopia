Installation
===========

Start by making a new conda environment:

.. code-block:: bash

    $ conda create --name my-mutopia -c conda-forge -y python=3.11
    $ conda activate my-mutopia

Next, install the package from github:

.. code-block:: bash

    $ pip install git+https://github.com/AllenWLynch/Mutopia.git


Mutopia model training
==========================

Let's say you have built a G-tensor with some data found at `gtensor.nc`. The first step in learning a model on that data is to perform hyperparameter optimization.
Split the data into train and test sets, listing the contigs you would like to leave out for test scoring (in this case, chr2):

.. code-block:: bash
    
    $ mutopia split gtensor.nc chr2


This will create two new files, `gtensor.train.nc` and `gtensor.test.nc`. Let's first make sure we can run a model on the data:

.. code-block:: bash

    $ mutopia train \
        model.pth \
        -train gtensor.train.nc \
        -test gtensor.test.nc \
        -k 10 \
        -bsub 0.125 \
        -lsub 0.125 \
        -@ 5

The command above lists the most important parameters used for training, but you can see all of them using `mutopia train --help`. The parameters above are:

* `model.pth` is the name of the model to be saved
* `-train` is the training data
* `-test` is the test data
* `-k` is the number of components to fit
* `-bsub` is the subsampling rate of samples during training (analogous to batch size)
* `-lsub` is the subsampling rate of loci during training
* `-@` is the number of threads to use

Run this for a few epochs to confirm the test score is *increasing*. If the epochs are taking too long, 
decrease `-lsub` and `-bsub`. If you decrease them too far, model training will be unstable and the 
test score will oscillate. 

An additional paramter that may be useful is `-init`, which initializes a component with some 
precomputed signature stored in the Mutopia database. For example, to initialize components with the 
"Motif Out5p" signatures `MO1` and `MO2`, use:

.. code-block:: bash

    $ mutopia train \
        model.pth \
        -train gtensor.train.nc \
        -test gtensor.test.nc \
        -k 10 \
        -bsub 0.125 \
        -lsub 0.125 \
        -@ 5 \
        -init MO1 \
        -init MO2

Note that you pass `-init` multiple times to initialize multiple components.

Once the model appears to be training well, you can start hyperparameter optimization. First, set up a "study", passing in the 
parameters you would like to use:

.. code-block:: bash

    $mkdir -p models/
    $ mutopia study create \
        tutorial.study.01 \
        -train gtensor.train.nc \
        -test gtensor.test.nc \
        -e \
        -bsub 0.125 \
        -lsub 0.125 \
        --save-model \
        --output-dir models/
        -init MO1 \
        -init MO2

Since the number of components will be tuned, we can omit that here. Make sure to set your `-bsub` and `-lsub` values to be the same as you used during setup.
Finally, note the new parameter `-e`, which stands for "extensive". You can pass additional `e`'s to make the study more extensive and tune more parameters, e.g. `-eee`.
For most purposes, `-ee` is sufficient. The options `--save-model` and `--output-dir` are used to save the model after training, and the output directory for the study, respectively.

Finally, to run a trial, use:

.. code-block:: bash

    $ mutopia study run tutorial.study.01 -@ 5

Which will train a model with randomly sampled hyperparameters and save its performance. Launch multiple trials across many slurm jobs to parallelize the search.
Make sure to tinker with the `--mem-per-cpu` and `--time` parameters to make sure your jobs are not killed before they finish.

.. code-block:: bash
    
    $ study=tutorial.study.01
    $ sbatch \
        --array=1-25 \
        --job-name=$study \
        --output=logs/$study/%A/%a.out \
        --partition=short,park \
        --time=24:00:00 \
        --ntasks=1 \
        --cpus-per-task=5 \
        --mem-per-cpu=2G \
        mutopia study run $study -@ 5

To check how training is going, use:

.. code-block:: bash

    $ mutopia study summary tutorial.study.01

which prints out a table of the trials, with the best performing trials at the top. You can use the column "user_attrs_model_path"
to find the saved checkpoint for each trial. As a final aside, you can run `mutopia study ls` to list all of your ongoing studies 
in case you've forgotten the name!

Let's assume that tuning is completed, and through `mutopia study summary` you found a model that showed good performance. To 
inspect the extracted signatures, we have to switch over to the python API.


Mutopia model analysis
==========================

Once you have a model you're happy with, you can use it to annotate a G-tensor. First, make a copy of your data, then run 
the `mutopia predict` command:

.. code-block:: bash

    $ cp gtensor.nc gtensor.annotated.nc
    $ mutopia predict path/to/model.pth gtensor.annotated.nc -@ 5

This will pre-load the data with information about the discovered components, like the contribution of each component to each sample. You can also
generate a report which shows the spectrum of each component:

.. code-block:: bash

    $ mutopia report path/to/model.pth report.pdf

Finally, you can use the model to annotate observations from new samples (this only works for SBS data for now). Run the `gtensor-sbs annotate` command,
and provide paths to a trained model, an annotated G-tensor, and a VCF file:

.. code-block:: bash

    $ gtensor-sbs annotate \
        path/to/model.pth \
        gtensor.annotated.nc \
        path/to/sample.bcf \
        --pass-only \
        -m mutation_rate.bedgraph.gz \
        | bcftools view -Oz > annotated.vcf.gz

Note that by omitting `--weight-col`, you are weighting each observation equally when solving for the sample-level contributions (which is 
usually the desired behavior). The "mutation_rate.bedgraph.gz" file is used for clustering. If clustering is not desired, pass the `--no-cluster` flag.

This command adds the INFO columns "logp_{component}" to the VCF file, which contains the posterior log-probability that the component generated the observation. 

Mutopia API usage
==========================

Expansion coming soon!

First, let's load the model and the data:

.. code-block:: python

    import mutopia as mu

    model = mu.load_model("path/to/model.pth")
    data = mu.load_data("gtensor.test.nc") # the training data may be quite large, it's often easier to work with the test data


