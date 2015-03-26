#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
# Run all examples in this directory sequentially, while collecting timing and
# performance information.  Stats against prior runs are compared as well as
# appended to the named file.
THIS_DIR=$(cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)
OUT_FILE="${HOME}/.nervana/all_example_results.tsv"
LOG_FILE="${HOME}/.nervana/all_example_results.log"
NEON_EXE="${THIS_DIR}/../bin/neon"
NEON_OPTS="-r 0 -o ${OUT_FILE}"  # non-distributed, CPU backend

mkdir -p "$(dirname $OUT_FILE)"
mkdir -p "$(dirname $LOG_FILE)"
make -C ${THIS_DIR}/.. build > /dev/null 2>&1  # ensure build is up to date
for dir in autoencoder balance convnet mlp recurrent
do
  for f in ${dir}/*.yaml
  do
    if [[ "$f" == *"hyperopt"* ]]
    then
      # skip hyperopt examples since they need to be run by bin/hyperopt
      continue
    fi
    echo "$(date) - Running: $f ..." | tee -a "$LOG_FILE"
    PYTHONPATH="${THIS_DIR}/..:${PYTHONPATH}" $NEON_EXE $NEON_OPTS "$f" \
        >> "$LOG_FILE" 2>&1
    if [ $? -eq 0 ]
    then
      PYTHONPATH="${THIS_DIR}/..:${PYTHONPATH}" python - <<END
from neon.util.metrics import compare_metrics
compare_metrics("$OUT_FILE", "$f")
END
    else
      echo "problems running $f"
    fi
    echo "" >> "$LOG_FILE"
  done
done
