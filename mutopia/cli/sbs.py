import mutopia as mu
from mutopia.modalities._sbs_clustering import *
from mutopia.corpus import disk_interface as disk
from ..genome_utils.bed12_utils import stream_bed12
import click
from functools import partial
from typing import *

SBS = mu.Modality.SBS.get_config()

@click.group("SBS commands")
def sbs():
    pass

@sbs.command("background-rate")
@click.argument("vcf_files", nargs=-1)
@click.option(
    '-g',
    '--genome-file',
    required=True,
    type=click.Path(exists=True),
    help='The reference genome fasta file.'
)
@click.option('-chr', '--chr-prefix', default='')
@click.option(
    '--pass-only/--no-pass-only',
    type=bool,
    default=True,
    help='Whether to only ingest passing mutations',
)
def marginal_rate(
    vcf_files : list,
    genome_file : str,
    chr_prefix='',
    pass_only=True,
):
    get_marginal_mutation_rate(
        genome_file,
        *vcf_files,
        chr_prefix=chr_prefix,
        pass_only=pass_only,
    )


@sbs.command("cluster")
@click.argument("vcf_file")
@click.option(
    '-g',
    '--mutation-rate-bedgraph',
    required=True,
    type=click.Path(exists=True),
    help='The mutation rate bedgraph file.'
)
@click.option('-chr', '--chr-prefix', default='')
@click.option(
    '--pass-only/--no-pass-only',
    type=bool,
    default=True,
    help='Whether to only ingest passing mutations',
)
@click.option(
    '-name',
    '--sample-name',
    type=str,
    default=None,
    help='VCFS ONLY: Name of sample in multi-sample VCF.',
)
def _cluster_vcfs(
    sample_name=None,
    pass_only=True,
    *,
    vcf_file,
    chr_prefix,
    mutation_rate_bedgraph,
):
    query_fn = partial(
        stream_passed_SNVs,
        sample=sample_name,
        pass_only=pass_only,
        chr_prefix=chr_prefix,
    )

    cluster_vcf(
        mutation_rate_bedgraph=mutation_rate_bedgraph,
        query_fn=partial(query_fn, vcf_file),
        vcf_file=vcf_file,
        chr_prefix=chr_prefix,
    )


@sbs.command("unstack")
@click.argument("input")
@click.argument("output")
def unstack(
    input,
    output,
):
    disk.write_dataset(
        SBS.unstack(disk.load_dataset(input)),
        output,
    )

@sbs.command("annotate")
@click.argument(
    'model',
    type=click.Path(exists=True),
)
@click.argument(
    'dataset',
    type=click.Path(exists=True),
)
@click.argument(
    'sample_file',
    type=click.Path(exists=True),
)
@click.option(
    '-m',
    '--mutation-rate-file',
    type=str,
    help='VCFS ONLY: File containing mutation rates',
)
@click.option(
    '-chr',
    '--chr-prefix',
    type=str,
    default='',
    help='VCFS ONLY: Prefix to add to chromosome names',
)
@click.option(
    '--pass-only/--no-pass-only',
    type=bool,
    default=True,
    help='VCFS ONLY: Whether to only ingest passing mutations',
)
@click.option(
    '-w',
    '--weight-col',
    type=str,
    default=None,
    help='VCFS ONLY: Column to use for mutation weights',
)
@click.option(
    '-name',
    '--sample-name',
    type=str,
    default=None,
    help='VCFS ONLY: Name of sample in multi-sample VCF.',
)
@click.option(
    '--cluster/--no-cluster',
    type=bool,
    default=True,
    help='VCFS ONLY: Whether to cluster the mutations in the sample, requires a mutation rate file.',
)
@click.option(
    '--skip-sort/--no-skip-sort',
    type=bool,
    default=False,
    help='Whether to skip sorting the VCF file. This will fail if the VCF is not in lexigraphical sorted order (chr1, chr10, ...).',
)
@click.option(
    '-fa',
    '--fasta',
    type=click.Path(exists=True),
    default=None,
    help='The reference genome fasta file.',
)
def annotate(
    dataset : str,
    model : str,
    sample_file : str,
    fasta=None,
    **kwargs,
):
    
    attrs = disk.read_attrs(dataset)
    regions_file=disk.fetch_regions_path(dataset)
    num_regions = sum(1 for _ in stream_bed12(regions_file))
    locus_coords = disk.read_coords(dataset)['locus']

    corpus = disk.load_dataset(dataset, with_samples=False, with_state=True)
    model = mu.load_model(model)

    fasta = fasta or attrs['fasta_file']
    if not os.path.exists(fasta):
        raise click.FileError(f'No such file exists: {fasta}, provide a valid fasta file using the `--fasta/-fa` argument.')
    
    if not mu.model.CS.has_corpusstate(corpus):
        raise ValueError(f'The provided G-Tensor is not annotated. Please run `mutopia predict <model> {corpus.attrs["filename"]}` with a trained model to annotate the corpus.')
        
    SBS.annotate_mutations(
        model,
        corpus,
        sample_file,
        **kwargs,
        regions_file=regions_file,
        locus_dim=num_regions,
        locus_coords=locus_coords,
        fasta_file=fasta,
    )


@sbs.command("marginal-ll")
@click.argument(
    'model',
    type=click.Path(exists=True),
)
@click.argument(
    'dataset',
    type=click.Path(exists=True),
)
@click.argument(
    'sample_file',
    type=click.Path(exists=True),
)
@click.option(
    '-m',
    '--mutation-rate-file',
    type=str,
    help='VCFS ONLY: File containing mutation rates',
)
@click.option(
    '-chr',
    '--chr-prefix',
    type=str,
    default='',
    help='VCFS ONLY: Prefix to add to chromosome names',
)
@click.option(
    '--pass-only/--no-pass-only',
    type=bool,
    default=True,
    help='VCFS ONLY: Whether to only ingest passing mutations',
)
@click.option(
    '-w',
    '--weight-col',
    type=str,
    default=None,
    help='VCFS ONLY: Column to use for mutation weights',
)
@click.option(
    '-name',
    '--sample-name',
    type=str,
    default=None,
    help='VCFS ONLY: Name of sample in multi-sample VCF.',
)
@click.option(
    '--cluster/--no-cluster',
    type=bool,
    default=True,
    help='VCFS ONLY: Whether to cluster the mutations in the sample, requires a mutation rate file.',
)
@click.option(
    '--skip-sort/--no-skip-sort',
    type=bool,
    default=False,
    help='Whether to skip sorting the VCF file. This will fail if the VCF is not in lexigraphical sorted order (chr1, chr10, ...).',
)
@click.option(
    '-fa',
    '--fasta',
    type=click.Path(exists=True),
    default=None,
    help='The reference genome fasta file.',
)
@click.option(
    '-alpha',
    '--alpha',
    type=str,
    default=None,
    help='Use this prior in the model.'
)
@click.option(
    '-@',
    '--threads',
    type=click.IntRange(1, 64),
    default=1,
    help='Number of threads to use.',
)
@click.option(
    '-s',
    '--sample-steps',
    type=click.IntRange(1, 10000000000),
    default=1000,
    help='Number of draws to take from gibbs sampler.',
)
@click.option(
    '-r',
    '--reps',
    type=click.IntRange(1, 1000),
    default=100,
    help='Number of runs of the sampler to average.',
)
def marginal_ll(
    dataset : str,
    model : str,
    sample_file : str,
    fasta : Union[str, None] = None,
    threads : int = 1,
    sample_steps : int = 10000,
    reps : int = 10,
    *,
    alpha : Union[str, None] = None,
    **ingest_kwargs,
):
    
    model : mu.model.Model = mu.load_model(model)
    # parse the alpha
    if not alpha is None:
        try:
            alpha = list(map(float, alpha.split(',')))
        except ValueError:
            raise click.BadParameter('Alphas must be a list of comma separated floats.')
        
        if not len(alpha) == model.n_components:
            raise click.BadParameter(f'Alphas must be of length {model.n_components}.')

    attrs = disk.read_attrs(dataset)
    coords = disk.read_coords(dataset)
    corpus = disk.load_dataset(dataset, with_samples=False, with_state=True)

    fasta = fasta or attrs['fasta_file']
    if not os.path.exists(fasta):
        raise click.FileError(f'No such file exists: {fasta}, provide a valid fasta file using the `--fasta/-fa` argument.')
    
    if not mu.model.CS.has_corpusstate(corpus):
        raise ValueError(f'The provided G-Tensor is not annotated. Please run `mutopia predict <model> {corpus.attrs["filename"]}` with a trained model to annotate the corpus.')
    
    regions_file=disk.fetch_regions_path(dataset)
    num_regions = sum(1 for _ in stream_bed12(regions_file))
        
    logger.info("Ingesting mutations ...")
    _, sample = mu.SBS.ingest_uncollaposed(
        sample_file,
        **ingest_kwargs,
        fasta_file=fasta,
        regions_file=regions_file,
        locus_dim=num_regions,
    )

    sample = sample.isel(locus=coords['locus'])

    model_state = model.model_state_

    ll, est_var = (
        model_state
        .locals_model
        .marginal_ll_sample(
            sample,
            corpus,
            model_state,
            alpha=alpha,
            threads=threads,
            sample_steps=sample_steps,
            reps=reps,
        )
    )

    print(ll, est_var, sep='\t', file=sys.stdout)
