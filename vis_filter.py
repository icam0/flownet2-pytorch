import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import torch
import numpy as np
import mmcv
from scipy.misc import imsave

from utils import flow_utils, tools, gpu_selection, get_layers,get_crop,HookClass
import datasets
import losses as losses_flow
import models as flownet_models
import argparse, os, sys, subprocess
from viz_toolkit.layer_activation_with_guided_backprop import GuidedBackprop
from viz_toolkit.cnn_layer_visualization import CNNLayerVisualization
from viz_toolkit import misc_functions
import warnings
import seaborn as sns

warnings.filterwarnings("ignore")

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
tools.add_arguments_for_module(parser, datasets, argument_for_class='inference_dataset', default='MpiSintelClean',
                               skip_params=['is_cropped'],
                               parameter_defaults={'root': '../MPI-Sintel/training',
                                                   'replicates': 1,
                                                   'norm_og': True}) #this activates the original normalization
parser.add_argument('--crop_size', type=int, nargs='+', default = [256, 256], help="Spatial dimension to crop training samples for training")
parser.add_argument('--inference_size', type=int, nargs='+', default = [-1,-1], help='spatial size divisible by 64. default (-1,-1) - largest possible valid size would be used')
parser.add_argument('--number_workers', '-nw', '--num_workers', type=int, default=8)
parser.add_argument('--number_gpus', '-ng', type=int, default=-1, help='number of GPUs to use')
parser.add_argument('--inference_n_batches', type=int, default=-1)
args = parser.parse_args()
args.cuda = True
args.model_class = tools.module_to_dict(flownet_models)[args.model]
args.loss_class = tools.module_to_dict(losses_flow)[args.loss]
args.inference_dataset_class = tools.module_to_dict(datasets)[args.inference_dataset]
args.effective_number_workers = args.number_workers * args.number_gpus
gpuargs = {'num_workers': args.effective_number_workers,
           'pin_memory': True,
           'drop_last': True} if args.cuda else {}
inf_gpuargs = gpuargs.copy()
inf_gpuargs['num_workers'] = args.number_workers

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

layers_flat = get_layers.remove_sequential(model)
layers_flat[20], layers_flat[28] = layers_flat[28], layers_flat[20] #Swap the layers so we can directly loop until predict_flow 6
layers_flat = layers_flat[0:21]

filters_nr = [64,128,256,256,512,512,512,512,1024,1024,1024]
recep_field = [7, 15, 31, 47, 63, 95, 127, 191, 255, 383,511]
stride_power = [1,2,3,3,4,4,5,5,6,6,6]
layer_names = ['conv1','conv2','conv3','conv3_1','conv4','conv4_1','conv5','conv5_1','conv6','conv6_1','predict_flow6']
cnn_layer_idx = 5 #always pick the relu one
layer_name = layer_names[cnn_layer_idx]
#filters = [0,87,93,130,157,181,227,239]
filters = [39,122,123,186,189,222,223,253]
inp_height = 31
inp_width = 31

fig,axes = plt.subplots(2,len(filters),figsize=(12.,8.))
recep_field_size = recep_field[cnn_layer_idx]

for i,filter in enumerate(filters):
    layer_vis = CNNLayerVisualization(layers_flat,cnn_layer_idx, filter,inp_height,inp_width,color=False,num_iter=20)
    # Layer visualization with pytorch hooks
    [img0,img1] = layer_vis.visualise_layer_with_hooks()
    if filter == 123:
        fig1,ax1 = plt.subplots(1,1)
        sns.distplot(img0.flatten(),ax=ax1,label='img0')
        sns.distplot(img1.flatten(), ax=ax1, label='img1')
        ax1.legend()
        fig1.show()
    axes[0,i].imshow(img0,cmap='gray')
    axes[1,i].imshow(img1,cmap='gray')
    axes[0,i].set_title('Filter %i'%(filter))

axes[0,0].set_ylabel('im0',rotation=0.,labelpad=20)
axes[1,0].set_ylabel('im1',rotation=0.,labelpad=20)
fig.suptitle('Layer %s'%(layer_names[cnn_layer_idx//2]))

for i, ax in enumerate(axes.flatten()):
    ax.set_xticks([])
    ax.set_yticks([])

#fig.suptitle('Layer %i, upscaling steps=%i, blur=%i, filter_sjze=%i,recep_size=%i' %(cnn_layer,upscaling_steps,blur,img1.shape[0],recep_field_size))
# save= False
# if save:
#     fig.savefig('./plots/layer%i-actviz.pdf'%(cnn_layer))
plt.show()






