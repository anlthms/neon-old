#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
# Run all examples in this directory consecutively and with timing information
# appended.
THIS_DIR=$(cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)
OUT_FILE="${THIS_DIR}/all_example_results.txt"
NEON_EXE="${THIS_DIR}/../bin/neon"
NEON_OPTS="-r 1"  # non-distributed, CPU backend

make -C ${THIS_DIR}/.. build  # ensure the build is up to date first
echo "" > "$OUT_FILE"
for dir in autoencoder balance convnet mlp recurrent
do
  for f in ${dir}/*.yaml
  do
    echo "Running: $f" >> "$OUT_FILE"
    (time PYTHONPATH="${THIS_DIR}/..:${PYTHONPATH}" $NEON_EXE $NEON_OPTS "$f")\
     >> "$OUT_FILE" 2>&1
    echo -e "\n\n\n" >> "$OUT_FILE"
  done
done
