#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
# Run all examples in this directory consecutively and with timing information
# appended.
OUT_FILE="./all_example_results.txt"
NEON_EXE="../bin/neon"

make -C .. build  # ensure the build is up to date first
echo "" > "$OUT_FILE"
for f in *.yaml
do
  echo "Running: $f" >> "$OUT_FILE"
  (time PYTHONPATH=".." $NEON_EXE "$f") >> "$OUT_FILE" 2>&1
  echo -e "\n\n\n" >> "$OUT_FILE"
done


