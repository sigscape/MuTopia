#!/bin/bash

function process_expression() {
  local file="${1}"

  ( 
    set -o pipefail
    awk -F '\t' 'NR>1 { print $1 "\t" $10 }' "$file" \
    | gtensor utils make-expression-bedfile
  )
}

if [ $# -ne 2 ]; then
    echo "Usage: $0 <input:url> <output:bed.gz>"
    exit 1
fi

process_expression "$1" \
    | LC_COLLATE=C sort -k1,1 -k2,2n \
    | bedtools merge -i - -s -c 5,6 -o collapse \
    | LC_COLLATE=C sort -k1,1 -k2,2n \
    | awk -v OFS="\t" '{
        split($4, ex, ",");
        split($5, strand, ",");
        max_ex = ex[1];
        max_strand = strand[1];
        for (i = 2; i <= length(ex); i++) {
            if (ex[i] > max_ex) {
                max_ex = ex[i];
                max_strand = strand[i];
            }
        }
        print $1,$2,$3,max_strand;
    }' \
    | LC_COLLATE=C sort -k1,1 -k2,2n > "$2"