import math
import numpy as np
from scipy import integrate

import anytree
from anytree import NodeMixin


class Convolution(NodeMixin):
    def __init__(self, s, k, p, i=None, parent=None, children=None):
        if parent:
            i = parent.o
        o = math.floor((i-k+2*p)/s)+1
        self.name = str(o)
        self.o = int(o)
        self.s = int(s)
        self.k = int(k)
        self.p = int(p)
        self.parent = parent
        if children:
            self.children = children


def generate_dimension_tree(input_dimension, target_dimension, depth):
    s_space = [1, 2, 3]
    k_space = [5] # restricting to only odd values due to padding
    p_space = [(lambda x: 0), (lambda x: math.floor(x / 2))]

    root_node = Convolution(1, 0, 0, i=input_dimension)
    root_node.o = input_dimension
    layers = [[root_node]]
    for layer in range(depth):
        layer_nodes = []
        for parent_node in layers[layer]:
            for s in s_space:
                  for k in k_space:
                      for p in p_space:
                        conv = Convolution(s, k, p(k), i=parent_node.o)
                        if conv.o <= input_dimension and conv.o >= target_dimension:
                          if (
                              (layer == depth - 1
                              and conv.o == target_dimension)
                              or layer < depth - 1
                          ):
                            conv.parent = parent_node
                            layer_nodes.append(conv)
        layers.append(layer_nodes)

    return layers


def integrate_dimension_profile(dimension_profile):
    return integrate.simps(dimension_profile)


def dimension_profile_kurtosis(
    dimension_profile, initial_dimension, target_dimension, depth
):
    a = target_dimension * depth
    b = initial_dimension * depth
    m = (a + b) / 2
    d = (b - a) / 2
    integral = integrate_dimension_profile(dimension_profile)
    kurtosis = (integral - m) / d
    return kurtosis


def get_sorted_dimension_profiles_with_kurtosis(
    layers, initial_dimension, target_dimension, depth
):
    downwards = 2
    root_node = layers[0][0]
    leaf_nodes_w_target_dim = anytree.search.findall_by_attr(
        root_node, str(target_dimension)
    )

    w = anytree.walker.Walker()
    paths_from_root_to_leaf_nodes = [
        w.walk(root_node, leaf) for leaf in leaf_nodes_w_target_dim
    ]

    dimension_profiles = [
        ([initial_dimension] + [layer.o for layer in path[downwards]], path[downwards])
        for path in paths_from_root_to_leaf_nodes
    ]
    dimension_profiles = list(
        filter(lambda x: len(x[0]) == depth + 1, dimension_profiles)
    )

    dimension_profiles_kurtosis = [
        (
            dimension_profile_kurtosis(
                dimension_profile[0], initial_dimension, target_dimension, depth
            ),
            dimension_profile[1],
            dimension_profile[0]
        )
        for dimension_profile in dimension_profiles
    ]

    sorted_dimension_profiles_kurtosis = sorted(
        dimension_profiles_kurtosis, key=(lambda x: x[0])
    )

    return sorted_dimension_profiles_kurtosis


def get_dimension_profile_with_closest_kurtosis(
    kurtosis, sorted_dimension_profiles_kurtosis
):
    sorted_kurtosis = [
        dimension_profile[0] for dimension_profile in sorted_dimension_profiles_kurtosis
    ]
    sorted_kurtosis = np.array(sorted_kurtosis)
    idx = sorted_kurtosis.searchsorted(kurtosis)
    idx = np.clip(idx, 1, len(sorted_kurtosis) - 1)
    left = sorted_kurtosis[idx - 1]
    right = sorted_kurtosis[idx]
    idx -= kurtosis - left < right - kurtosis
    return sorted_dimension_profiles_kurtosis[idx]


def evaluate_dimensions_per_layer(kurtosis, initial_dimension, target_dimension, depth):
    """
    Given a kurtosis value, input tensor dimension, output tensor dimensions and depth of the generator,
    calculates a dimension profile, ie the dimension of each layer of the generator.

    kurtosis: value between -1 and 1 that determines the morphology of dimensions progression ("exponential", "linear", "logarithmic")
    initial_dimension: integer, dimension of the reshaped dense layer, height/width of the input tensor
    target_dimension: integer, dimension of the output tensor
    depth: count of deconvolution layers

    Returns a list of nodes, ie Convolution objects
    """

    layers = generate_dimension_tree(initial_dimension, target_dimension, depth)
    sorted_dimension_profiles_kurtosis = get_sorted_dimension_profiles_with_kurtosis(
        layers, initial_dimension, target_dimension, depth
    )
    dimension_profile = get_dimension_profile_with_closest_kurtosis(
        kurtosis, sorted_dimension_profiles_kurtosis
    )
    layer_nodes = dimension_profile[1]
    return layer_nodes