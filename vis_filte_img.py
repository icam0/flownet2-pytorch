import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
import mmcv
import seaborn as sns
import time
import torch
import torch.nn as nn

from utils import flow_utils, tools, gpu_selection
import losses as losses_flow
import models as flownet_models
import argparse, os, sys, subprocess
from viz_toolkit.cnn_layer_visualization import CNNLayerVisualization
from viz_toolkit.layer_activation_with_guided_backprop import GuidedBackprop

#select the GPU
free_gpu_id = gpu_selection.get_freer_gpu()
print('Selecting GPU %i' %(free_gpu_id))
torch.cuda.set_device(free_gpu_id)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--resume', default='./checkpoints/FlowNet2-S_checkpoint.pth.tar', type=str, metavar='PATH')
tools.add_arguments_for_module(parser, losses_flow, argument_for_class='loss', default='L2Loss')
tools.add_arguments_for_module(parser, flownet_models, argument_for_class='model', default='FlowNet2S')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument("--rgb_max", type=float, default = 255.)
args = parser.parse_args()
args.model_class = tools.module_to_dict(flownet_models)[args.model]
args.loss_class = tools.module_to_dict(losses_flow)[args.loss]

# Load the model and loss
kwargs = tools.kwargs_from_args(args, 'model')
model = args.model_class(args, **kwargs)
model = model.cuda()
torch.cuda.manual_seed(args.seed)

# Load previous checkpoint
checkpoint = torch.load(args.resume)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
model.zero_grad()

all_layers = []
def remove_sequential(network):
    for layer in network.children():
        if type(layer) == nn.Sequential:  # if sequential layer, apply recursively to layers in sequential layer
            remove_sequential(layer)
        if list(layer.children()) == []:  # if leaf node, add it to list
            all_layers.append(layer)

remove_sequential(model)
all_layers[20], all_layers[28] = all_layers[28], all_layers[20]

recep_field = [7, 15, 31, 47, 63, 95, 127, 191, 255, 383,511]
filters_nr = [64,128,256,256,512,512,512,512,1024,1024,1024]

cnn_layer = 20

im0 = imread('./frame1.png').astype(np.float32)
im1 = imread('./frame2.png').astype(np.float32)
im0_mean = np.array((0.45014125, 0.4320596, 0.4114511))
im1_mean = np.array((0.44855332, 0.43102142,0.41060242))

# switch between these 2
im0_norm = (im0/255.)-im0_mean
im1_norm = (im1/255.)-im1_mean
images = np.concatenate((im0_norm,im1_norm),axis=2).transpose(2,0,1).reshape(1,6,384,512)
images = torch.tensor(images.astype(np.float32), requires_grad=True,device='cuda')

all_layers = all_layers[0:21]

cnn_layer = 15
filter_pos = 298
xy_pos = [6,7]
GBP = GuidedBackprop(all_layers)
# Get gradients
guided_grads = GBP.generate_gradients(images, None, cnn_layer, filter_pos,xy_pos)

guided_grads = guided_grads - guided_grads.min()
guided_grads /= guided_grads.max()

fig,axes = plt.subplots(2,2,)
axes[0,0].imshow(guided_grads[0:3,:,:].transpose(1,2,0))
axes[0,1].imshow(guided_grads[3:,:,:].transpose(1,2,0))

axes[1,0].imshow(im0/255.)
axes[1,1].imshow(im1/255.)

plt.show()

# visualisation = {}
# def hook_fn(m, i, o):
#     visualisation[m] = o
# def get_all_layers(net):
#     for name, layer in net._modules.items():
#         if isinstance(layer, nn.Sequential):
#             get_all_layers(layer)
#         else:
#             layer.register_forward_hook(hook_fn)
# get_all_layers(model_and_loss.model)
# out = model_and_loss.model(images)
# visualisation.keys()
# layers = [1,3,5,7,9,11,13,15,17,19]
# fig,axes = plt.subplots(len(layers),1,figsize=(8.,12.))
# for i,layer in enumerate(layers):
#     act = visualisation[list(visualisation.keys())[layer]].cpu().data.numpy()[0]
#     axes[i].plot(np.arange(len(act)),act.max(axis=(1,2)))
#     axes[i].set_title('Layer %i'%(layer))
#     loc = unravel_index(act[1].argmax(), act[1].shape)
# plt.show()







