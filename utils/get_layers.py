import torch.nn as nn

layers_flat = []
def remove_sequential(network):

    for layer in network.children():
        if type(layer) == nn.Sequential:  # if sequential layer, apply recursively to layers in sequential layer
            remove_sequential(layer)
        if list(layer.children()) == []:  # if leaf node, add it to list
            layers_flat.append(layer)
    return layers_flat
