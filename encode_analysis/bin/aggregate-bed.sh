#!/bin/bash

# Usage: aggregate-bed.sh <bin_size> <genome_file> <input_bed_file>
# This script aggregates bed file values into fixed-size bins.
# make it so that this script exits on error
set -e
BIN_SIZE=$1
GENOME_FILE=$2
INPUT_FILE=$3

bedtools makewindows -g $GENOME_FILE -w $BIN_SIZE | \
    LC_COLLATE=C sort -k1,1 -k2,2n | \
    bedtools map -a - -b $INPUT_FILE -c 5 -o mean | \
    gzip