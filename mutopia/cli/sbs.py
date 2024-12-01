import click
import sys
import mutopia as mu
from mutopia.modalities._sbs_ingestion import annotate_mutations, featurize_annotated_mutations
from mutopia.modalities._sbs_clustering import transfer_annotations_to_vcf, get_marginal_mutation_rate
import mutopia.corpus.disk_interface as disk

SBS = mu.Modality.SBS.get_config()

@click.group("SBS commands")
def sbs():
    pass

@sbs.command("annotate-vcf")
@click.argument("vcf_file")
@click.option(
    '-fa',
    '--fasta-file',
    required=True,
    type=click.Path(exists=True),
    help='The reference genome fasta file.'
)
@click.option('-chr', '--chr-prefix', default='')
def annotate_vcf(
    vcf_file : str,
    fasta_file : str,
    chr_prefix='',
):
    
    mutations_df = annotate_mutations(
        vcf_file,
        fasta_file=fasta_file,
        chr_prefix=chr_prefix,
    )
    
    transfer_annotations_to_vcf(
        mutations_df,
        vcf_file=vcf_file,
        chr_prefix=chr_prefix,
        output=sys.stdout,
        description={
            'CONFIGURATION' : 'The mutation\'s sequence strand orientation',
            'CONTEXT' : 'The mutation\'s sequence context',
            'MUTATION' : 'The mutation\'s ALT code',
            'MUTATION_CODE' : 'The Cosmic mutation designation',
        }
    )


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



@sbs.command("add-annotated")
@click.argument(
    'dataset',
    type=click.Path(exists=True),
)
@click.argument(
    'sample_file',
    type=click.Path(exists=True),
)
@click.option(
    '-id',
    '--sample-id',
    type=str,
    required=True,
    help='ID to assign the sample',
)
@click.option(
    '-m',
    '--mutation-rate-file',
    type=str,
    required=True,
    help='File containing mutation rates',
)
@click.option(
    '-chr',
    '--chr-prefix',
    type=str,
    default='',
    help='Prefix to add to chromosome names',
)
@click.option(
    '--pass-only/--no-pass-only',
    type=bool,
    default=True,
    help='Whether to only ingest passing mutations',
)
@click.option(
    '-w',
    '--weight-col',
    type=str,
    help='Column to use for mutation weights',
)
@click.option(
    '-sw',
    '--sample-weight',
    type=click.FloatRange(0., 1000000, min_open=False),
    default=1.,
    help='Weight to assign to the sample',
)
@click.option(
    '-name',
    '--sample-name',
    type=str,
    default=None,
    help='Name of sample in multi-sample VCF.',
)
def add_annotated(
    dataset,
    sample_file,
    sample_id,
    chr_prefix='',
    pass_only=True,
    weight_col=None,
    mutation_rate_file=None,
    sample_weight=None,
    sample_name=None,
):
    
    coords, weights = \
        featurize_annotated_mutations(
            sample_file,
            disk.read_regions_file(dataset),
            chr_prefix=chr_prefix,
            pass_only=pass_only,
            weight_col=weight_col,
            mutation_rate_file=mutation_rate_file,
            sample_weight=sample_weight,
            sample_name=sample_name,
        )

    sample_arr = SBS._arr_to_xr(
        disk.read_dims(dataset),
        coords, 
        weights
    )

    disk.write_sample(
        dataset,
        sample_arr,
        sample_name=f'X/{sample_id}',
    )
