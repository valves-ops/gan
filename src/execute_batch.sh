#!/bin/bash

for i in {1..3}
do
    /bin/bash execute_experiment.sh mnist.capacity_scaling
done
# /bin/bash execute_experiment.sh mnist.depth_scaling
# /bin/bash execute_experiment.sh mnist.dim_prog_combinations
# /bin/bash execute_experiment.sh mnist.filter_prog_combinations
# /bin/bash execute_experiment.sh mnist.depth_combinations
# /bin/bash execute_experiment.sh mnist.kernel_combinations
