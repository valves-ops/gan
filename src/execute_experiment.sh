#!/bin/bash

# set -x

experiment=$1
echo "Experiment $experiment"
separator='/'
experiment_path=".${separator}experiments${separator}${experiment/./$separator}${separator}runs"
permutations_path=".${separator}experiments${separator}${experiment/./$separator}${separator}permutations"
echo "Experiment path: $experiment_path"
echo "Permutations path: $permutations_path"

gin_run_files=`ls ${experiment_path}${separator}*.gin`
gin_permutation_files=`ls ${permutations_path}${separator}*.gin`

echo "Run files: $gin_run_files"
echo "Permutation files: $gin_permutation_files"

for gin_permutation_file in $gin_permutation_files
do
    for gin_run_file_path in $gin_run_files
    do
        gin_run_file=$(basename "$gin_run_file_path")
        echo "Executing ${gin_run_file}"
        python3 main.py --experiment ${experiment} --permutation ${gin_permutation_file} --run ${gin_run_file}
    done
done