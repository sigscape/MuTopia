import click
import sys
import mutopia as mu
from mutopia.modalities.fragment_motif import parse_bamfile

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
