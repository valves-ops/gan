#!/bin/bash

experiment=$1
echo "Experiment $experiment"
separator='/'
experiment_path=".${separator}experiments${separator}${experiment/./$separator}${separator}runs"
echo "Experiment path: $experiment_path"

gin_run_files=`ls ${experiment_path}${separator}*.gin`
for gin_run_file_path in $gin_run_files
do
    gin_run_file=$(basename "$gin_run_file_path")
    echo "Executing ${gin_run_file}"
    python3 main.py --experiment ${experiment} --run ${gin_run_file}
done