#!/bin/bash

if [ $# -lt 1 ]; then
    echo "ARGS ERROR!"
    echo "  bash install.sh /path/to/pysot"
    exit 1
fi

set -e

pysot_path=$1

rsync -rav ./experiments/ $pysot_path/experiments/
rsync -rav ./pysot/ $pysot_path/pysot/
