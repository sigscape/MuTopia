import click
from typing import *
from functools import partial
import mutopia.ingestion as ingest
from mutopia.corpus import GTensor, write_dataset
import mutopia.corpus.disk_interface as disk
from ..modalities import Modality
from ..genome_utils.bed12_utils import stream_bed12
from ..utils import FeatureType, logger

##
# TODO:
# The exposures should be added like a feature.
# Provide a bed file bigwig file and use this to find the average or sum 
# within the regions.
## 

@click.group("G-tensor commands")
def ingestion():
    pass

@ingestion.command("create")
@click.argument(
    'cutout_regions',
    type=click.Path(exists=True),
    nargs=-1,
)
@click.option(
    '-n','--name',
    type=str,
    required=True,
    help='Name to assign the G-Tensor',
)
@click.option(
    '-o',
    '--output',
    type=click.Path(writable=True),
    required=True,
    help='Output file to write the G-Tensor to',
)
@click.option(
    '-dtype',
    '--dtype',
    type=click.Choice([x.value for x in Modality]),
    required=True,
    help='Modality to use for the G-Tensor',
)
@click.option(
    '-g',
    '--genome-file',
    type=click.Path(exists=True),
    required=True,
    help='Genome file to use for the regions',
)
@click.option(
    '-v',
    '--blacklist-file',
    type=click.Path(exists=True),
    required=True,
    help='File containing loci to exclude from the regions',
)
@click.option(
    '-fa',
    '--fasta-file',
    type=click.Path(exists=True),
    required=True,
    help='Fasta file to use for calculating context frequencies',
)
@click.option(
    '-s',
    '--region-size',
    type=click.IntRange(1,1000000),
    default=10000,
    help='Size of the regions to create',
)
@click.option(
    '-min',
    '--min-region-size',
    type=click.IntRange(1,1000000),
    default=25,
    help='Minimum size of the regions to create',
)
def create(
    *,
    name : str,
    dtype : str,
    output : str,
    genome_file,
    fasta_file,
    blacklist_file,
    cutout_regions : List[str]=[],
    min_region_size=25,
    region_size : int = 10000,
):
    logger.info('Creating genomic regions ...')
    regions_file = output + '.regions.bed'
    with open(regions_file, 'w') as r:
        ingest.make_regions(
            *cutout_regions,
            genome_file=genome_file,
            window_size=region_size,
            blacklist_file=blacklist_file,
            min_windowsize=min_region_size,
            output=r,
        )

    logger.info('Calculating context frequencies ...')
    
    modality = Modality(dtype).get_config()

    context_freqs = modality.get_context_frequencies(
        regions_file=regions_file,
        fasta_file=fasta_file,
    )

    logger.info('Formatting G-Tensor ...')

    chrom, start, end = list(zip(*(
        (s.chromosome, s.start, s.end)
        for s in stream_bed12(regions_file)
    )))

    gtensor = GTensor(
        modality,
        name=name,
        chrom=chrom,
        start=start,
        end=end,
        context_frequencies=context_freqs,
    )

    gtensor.attrs['regions_file'] = regions_file
    gtensor.attrs['genome_file'] = genome_file
    gtensor.attrs['fasta_file'] = fasta_file
    gtensor.attrs['blacklist_file'] = blacklist_file
    gtensor.attrs['region_size'] = region_size

    logger.info(
        f'Wrote G-Tensor to {output}, and accompanying regions file to {regions_file}\n'
        'Make sure not to leave the regions file behind if you move the G-Tensor!.'
    )
    
    write_dataset(gtensor, output)


@ingestion.group("feature")
def featurecmds():
    pass

@featurecmds.command("ingest-continuous")
@click.argument(
    'dataset',
    type=click.Path(exists=True),
)
@click.argument(
    'ingest_file',
    type=click.Path(exists=True),
)
@click.option(
    '-name',
    '--feature-name',
    type=str,
    required=True,
    help='Name to assign the feature',
)
@click.option(
    '-norm',
    '--normalization',
    type=click.Choice([x.value for x in FeatureType.continuous_types()]),
    default='power',
    help='Normalization to apply to the feature',
)
@click.option(
    '-g',
    '--group',
    default='all',
    type=str,
    help='Group to assign the feature to',
)
def ingest_continuous_feature(
    dataset : str,
    ingest_file : str,
    normalization : str ='power',
    group='all',
    *,
    feature_name : str,
):
    
    file_type = ingest.FileType.from_extension(ingest_file)
    
    if not file_type in (ingest.FileType.BEDGRAPH, ingest.FileType.BIGWIG):
        raise ValueError(f'File type {file_type} not supported for continuous feature ingestion.')
    
    if not FeatureType(normalization) in file_type.allowed_normalizations:
        raise ValueError(f'Normalization {normalization} not supported for file type {file_type}')

    corpus_attrs = disk.read_attrs(dataset)

    feature_vals = file_type.get_ingestion_fn()(
        ingest_file,
        corpus_attrs['regions_file'],
        genome_file=corpus_attrs['genome_file'],
        extend=corpus_attrs['region_size'],
    )

    disk.write_feature(
        dataset,
        feature_vals,
        group=group,
        name=feature_name,
        normalization=FeatureType(normalization),   
    )


@featurecmds.command("ingest-discrete")
@click.argument(
    'dataset',
    type=click.Path(exists=True),
)
@click.argument(
    'ingest_file',
    type=click.Path(exists=True),
)
@click.option(
    '-name',
    '--feature-name',
    type=str,
    required=True,
    help='Name to assign the feature',
)
@click.option(
    '--mesoscale/--no-mesoscale',
    default=True,
    help='Whether to treat the feature as mesoscale',
    type=bool,
    is_flag=True,
)
@click.option(
    '-null',
    '--null',
    type=str,
    default='none',
    help='Value to assign to null regions',
)
@click.option(
    '-col',
    '--column',
    type=int,
    default=4,
    help='Column in the bedfile to use for the class',
)
@click.option(
    '-p',
    '--class-priority',
    type=str,
    multiple=True,
    help='Priority of classes in the bedfile',
)
@click.option(
    '-g',
    '--group',
    default='all',
    type=str,
    help='Group to assign the feature to',
)
def ingest_discrete(
    ingest_file : str,
    group : str = 'all',
    mesoscale : bool = True,
    null : str = 'none',
    column : int = 4,
    class_priority : List[str] = [],
    *,
    dataset : str,
    feature_name : str,
):

    if not ingest.FileType.from_extension(ingest_file) == ingest.FileType.BED:
        raise ValueError('Discrete features must be ingested from Bed files.')
    
    corpus_attrs = disk.read_attrs(dataset)

    feature_vals = ingest.make_discrete_features(
        ingest_file,
        corpus_attrs['regions_file'],
        column=column,
        null=null,
        class_priority=class_priority if len(class_priority) > 0 else None,
    )

    disk.write_feature(
        dataset,
        feature_vals,
        group=group,
        name=feature_name,
        normalization=FeatureType.MESOSCALE if mesoscale else FeatureType.CATEGORICAL,
    )


@featurecmds.command("ingest-distance")
@click.argument(
    'dataset',
    type=click.Path(exists=True),
)
@click.argument(
    'ingest_file',
    type=click.Path(exists=True),
)
@click.option(
    '-name',
    '--feature-name',
    type=str,
    required=True,
    help='Name to assign the feature',
)
@click.option(
    '-g',
    '--group',
    default='all',
    type=str,
    help='Group to assign the feature to',
)
def ingest_distance(
    ingest_file : str,
    group : str = 'all',
    *,
    dataset : str,
    feature_name : str,
):
    if not ingest.FileType.from_extension(ingest_file) == ingest.FileType.BED:
        raise ValueError('Discrete features must be ingested from Bed files.')

    corpus_attrs = disk.read_attrs(dataset)

    (progress_between, distance_between) = \
        ingest.make_distance_features(
            ingest_file,
            corpus_attrs['regions_file'],
        )
    
    write_fn = partial(
        disk.write_feature,
        dataset,
        group=group,
        normalization=FeatureType.QUANTILE,
    )

    write_fn(
        progress_between,
        name=f'{feature_name}_positionBetween',
    )

    write_fn(
        distance_between,
        name=f'{feature_name}_distanceBetween',
    )



@featurecmds.command("list-features")
@click.argument(
    'dataset',
    type=click.Path(exists=True),
)
def list_features(
    dataset : str,
):
    feature_info = disk.list_features(dataset)
    
    if len(feature_info)==0:
        click.echo("No features found in the dataset.")
        return

    headers = ['Feature name', 'normalization', 'group']
    rows = [
        [feature_name] + [str(attrs.get(header, "")) for header in headers[1:]]
        for feature_name, attrs in feature_info.items()
        if bool(attrs['active'])
    ]

    col_widths = [max(len(str(item)) for item in col) for col in zip(*([headers] + rows))]

    def format_row(row):
        return " | ".join(f"{item:<{col_widths[i]}}" for i, item in enumerate(row))

    click.echo(format_row(headers))
    click.echo("-+-".join("-" * width for width in col_widths))
    for row in rows:
        click.echo(format_row(row))



@featurecmds.command("rm-features")
@click.argument(
    'dataset',
    type=click.Path(exists=True),
)
@click.argument(
    'feature_names',
    type=str,
    nargs=-1,
)
def rm_features(
    dataset : str,
    feature_names : List[str],
):
    for feature_name in feature_names:
        disk.rm_feature(
            dataset,
            feature_name,
        )
        click.echo(f'Removed feature: {feature_name}')


@ingestion.group("sample")
def samplecmds():
    pass

@samplecmds.command("ingest")
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
def ingest_sample(
    dataset : str,
    sample_file : str,
    sample_name : str,
    chr_prefix : str = '',
    pass_only : bool = True,
    weight_col : Union[None, str] = None,
    mutation_rate_file : Union[None, str] = None,
    sample_weight : Union[None, float] = 1.,
    *,
    sample_id : str,
):
    
    attrs = disk.read_attrs(dataset)
    dims = disk.read_dims(dataset)

    modality = Modality(attrs['dtype'].upper()).get_config()

    sample_arr = modality.ingest_observations(
        sample_file,
        regions_file=attrs['regions_file'],
        fasta_file=attrs['fasta_file'],
        chr_prefix=chr_prefix,
        pass_only=pass_only,
        weight_col=weight_col,
        mutation_rate_file=mutation_rate_file,
        sample_weight=sample_weight,
        sample_name=sample_name,
        dim_sizes=dims,
    )

    disk.write_sample(
        dataset,
        sample_arr,
        sample_name=f'X/{sample_id}',
    )


@samplecmds.command("rm-samples")
def rm_samples():
    pass


@samplecmds.command("list-samples")
def list_samples():
    pass
