import click
import sys
import mutopia as mu
from mutopia.modalities.fragment_motif import parse_bamfile
from mutopia.modalities.fragment_length import bam_to_fragments

@click.group("Fragment commands")
def fragments():
    pass

@fragments.command("parse-bam")
@click.argument("bam_file")
@click.argument("weight_tags", nargs=-1)
@click.option(
    '-o',
    '--output',
    default=sys.stdout,
    type=click.File('w'),
)
def _parse_bam(*, bam_file, weight_tags, output):
    parse_bamfile(bam_file, *weight_tags, output=output)


@fragments.command("bam-to-fragments")
@click.argument("bam_file")
@click.argument("weight_tags", nargs=-1)
@click.option(
    '-o',
    '--output',
    default=sys.stdout,
    type=click.File('w'),
)
def _bam_to_fragments(*, bam_file, weight_tags, output):
    bam_to_fragments(bam_file, *weight_tags, output=output)
