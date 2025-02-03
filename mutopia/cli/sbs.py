import mutopia as mu
from mutopia.modalities._sbs_clustering import *
from mutopia.corpus import disk_interface as disk
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
    sample_name : str = None,
    chr_prefix : str = '',
    pass_only : bool = True,
    weight_col : Union[None, str] = None,
    mutation_rate_file : Union[None, str] = None,
    sample_weight : Union[None, float] = 1.,
    weight_tags : List[str] = [],
    skip_sort : bool = False,
    cluster : bool = True,
    fasta=None,
):
    
    attrs = disk.read_attrs(dataset)
    corpus = disk.load_dataset(dataset, with_samples=False)
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
        sample_name=sample_name,
        chr_prefix=chr_prefix,
        pass_only=pass_only,
        weight_col=weight_col,
        mutation_rate_file=mutation_rate_file,
        sample_weight=sample_weight,
        weight_tags=weight_tags,
        skip_sort=skip_sort,
        cluster=cluster,
        regions_file=disk.fetch_regions_path(dataset),
        locus_dim=disk.read_dims(dataset)['locus'],
        fasta_file=fasta,
    )
