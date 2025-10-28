#!/usr/bin/env python3

import sys
import click
from gzip import open as gzopen
import tempfile
import subprocess

def bed_to_segments(
    bedfile,
    output=sys.stdout,
    width=2000,
    overhang=1,
    max_id=20, 
):

    segments=[]
    for line in bedfile:

        line = line.strip().split("\t")

        (chrom, start, end, name, value, strand) = line[:6]

        start = int(start); end = int(end)

        if strand == '+':
            segment_starts = list(range(start-overhang*width, end, width))
        if strand == '-':
            segment_starts = list(range(end+overhang*width, start, -width))

        for seg_id, segment_start in enumerate(segment_starts):

            seg_id -= overhang

            segment_start = max(segment_start, 0)
            segment_end = segment_start + width

            segments.append(
                (chrom, segment_start, segment_end, seg_id)
            )

            if seg_id>=max_id:
                break

    segments = sorted(segments, key=lambda x: (x[0], x[1]))

    for segment in segments:
        print(*segment, sep="\t", file=output)
    

def run(
    bedfile, 
    width=2000,
    max_id=100,
    overhang=1,
    output=sys.stdout
):
    
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_output:
        bed_to_segments(
            bedfile, 
            width=width, 
            max_id=max_id, 
            overhang=overhang,
            output=temp_output
        )
        temp_output.flush()
        temp_output.seek(0)

        subprocess.check_call(
            'sort -k1,1 -k2,2n {} | bedtools merge -i stdin -c 4 -o min -d -1 | LC_COLLATE=C sort -k1,1 -k2,2n'.format(temp_output.name),
            stdout=output,
            shell=True
        )

def main():
    import argparse
    import sys

    argparser = argparse.ArgumentParser(
        description="Generate tiled segments for genes from a BED file."
    )
    argparser.add_argument(
        "BEDFILE",
        type=argparse.FileType('r'),
        default=sys.stdin,
        help="Input BED file with gene annotations."
    )
    argparser.add_argument(
        "--width",
        type=int,
        default=2000,
        help="Width of each tile segment (default: 2000)."
    )
    argparser.add_argument(
        "--max_id",
        type=int,
        default=100,
        help="Maximum segment ID per gene (default: 100)."
    )
    argparser.add_argument(
        "--overhang",
        type=int,
        default=1,
        help="Number of overhang segments to include (default: 1)."
    )
    argparser.add_argument(
        "--output",
        type=click.File('w'),
        default=sys.stdout,
        help="Output file (default: stdout)."
    )

    args = argparser.parse_args()

    run(
        args.BEDFILE,
        width=args.width,
        max_id=args.max_id,
        overhang=args.overhang,
        output=args.output
    )

if __name__ == '__main__':
    main()
