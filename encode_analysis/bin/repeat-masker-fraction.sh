#!/bin/bash

# Usage: aggregate-bed.sh <bin_size> <input_fasta>
# This script determines the fraction of bases in a genome that are masked by RepeatMasker.
# makewindows -> nuc -> cut -> awk to count the number of lowercase bases (masked) vs total bases

set -e
BIN_SIZE=$1
GENOME_FILE=$2
FASTA_FILE=$3

bedtools makewindows -g $GENOME_FILE -w $BIN_SIZE | \
    LC_COLLATE=C sort -k1,1 -k2,2n | \
    bedtools nuc -fi $FASTA_FILE -bed - -seq | \
    cut -f1-3,13 | \
    awk -v OFS="\t" '{
        seq=$4
        total=length(seq)
        masked=gsub(/[a-zN]/, "", seq)
        fraction=masked/(total+1)
        print $1, $2, $3, fraction
    }' | gzip