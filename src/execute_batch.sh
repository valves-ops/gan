#!/bin/bash

# for i in {1..2}
# do
#     /bin/bash execute_experiment.sh mnist.depth_combinations
# done

# /bin/bash execute_experiment.sh mnist.capacity_scaling
# /bin/bash execute_experiment.sh mnist.dim_prog_combinations
# /bin/bash execute_experiment.sh mnist.filter_prog_combinations
/bin/bash execute_experiment.sh mnist.depth_combinations

/bin/bash execute_experiment.sh cifar.capacity_scaling
# /bin/bash execute_experiment.sh cifar.dim_prog_combinations
# /bin/bash execute_experiment.sh cifar.filter_prog_combinations
# /bin/bash execute_experiment.sh cifar.depth_combinations
