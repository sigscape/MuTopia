import click
import mutopia as mu
from mutopia.modalities._sbs_clustering import *
from mutopia.corpus import disk_interface as disk
from functools import partial

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