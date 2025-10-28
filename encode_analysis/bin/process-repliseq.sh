#!/bin/bash
set -e
if [ $# -ne 4 ]; then
    echo "Usage: $0 <chain_file> <new-genome-file> <input:bigwig> <output:bigwig>"
    exit 1
fi

chain_file=$1
new_genome=$2
input=$3
output=$4

bedgraph=$(mktemp)
lifted_bg=$(mktemp)

# 1. Convert bigwig to bedgraph
bigWigToBedGraph $input $bedgraph

# 2. LiftOver bedgraph
liftOver $bedgraph $chain_file $lifted_bg $lifted_bg.unmapped

sort -k1,1 -k2,2n $lifted_bg | \
    bedtools merge -i stdin -c 4 -o mean -d -1 | \
    LC_ALL=C sort -k1,1 -k2,2n \
    > $lifted_bg.sorted
    
mv $lifted_bg.sorted $lifted_bg

# 3. Convert bedgraph to bigwig
bedGraphToBigWig $lifted_bg $new_genome $output

rm $bedgraph $lifted_bg $lifted_bg.unmapped
