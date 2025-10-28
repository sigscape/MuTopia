#!/bin/bash

function process_expression() {
  local file="${1}"

  ( 
    set -o pipefail
    awk -F '\t' 'NR>1 { print $1 "\t" $10 }' "$file" \
    | gtensor utils make-expression-bedfile
  )
}

if [ $# -ne 1 ]; then
    echo "Usage: $0 <input:url> <output:tsv>"
    exit 1
fi

process_expression "$1" | LC_COLLATE=C sort -k1,1 -k2,2n