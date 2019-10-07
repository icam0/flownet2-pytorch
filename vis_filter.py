import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from utils import flow_utils, tools
import losses as losses_flow
import models as flownet_models
import argparse, os, sys, subprocess

import numpy as np
from viz_toolkit.cnn_layer_visualization import CNNLayerVisualization
import mmcv
from skimage.io import imread
import seaborn as sns
#sns.set()
import time

with torch.cuda.device(0):

    class ModelAndLoss(nn.Module):
        def __init__(self, args):
            super(ModelAndLoss, self).__init__()
            kwargs = tools.kwargs_from_args(args, 'model')
            self.model = args.model_class(args, **kwargs)
            kwargs = tools.kwargs_from_args(args, 'loss')
            self.loss = args.loss_class(args, **kwargs)

        def forward(self, data, target, inference=False ):
            output = self.model(data)
            loss_values = self.loss(output, target)
            if not inference :
                return loss_values
            else :
                return loss_values, output

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
    model_and_loss = ModelAndLoss(args)
    model_and_loss = model_and_loss.cuda()
    torch.cuda.manual_seed(args.seed)

    # Load previous checkpoint
    checkpoint = torch.load(args.resume)
    model_and_loss.model.load_state_dict(checkpoint['state_dict'])

    # Fully connected layer is not needed
    pretrained_model = model_and_loss.model
    pretrained_model.eval()

    all_layers = []
    def remove_sequential(network):
        for layer in network.children():
            if type(layer) == nn.Sequential:  # if sequential layer, apply recursively to layers in sequential layer
                remove_sequential(layer)
            if list(layer.children()) == []:  # if leaf node, add it to list
                all_layers.append(layer)

    remove_sequential(pretrained_model)

    all_layers[20], all_layers[28] = all_layers[28], all_layers[20]

    recep_field = [7, 15, 31, 47, 63, 95, 127, 191, 255, 383,511]
    filters_nr = [64,128,256,256,512,512,512,512,1024,1024,1024]

    cnn_layer = 9
    np.random.seed(2)
    filters = [158]#np.random.randint(0,filters_nr[cnn_layer//2],size=5)
    fig,axes = plt.subplots(3,len(filters),figsize=(12.,8.))
    recep_field_size = recep_field[cnn_layer//2]
    upscaling_steps = 1
    blur= 1

    # 14 layer 1,
    # 158 layer 9

    for i,filter_pos in enumerate(filters):
        layer_vis = CNNLayerVisualization(all_layers, cnn_layer, filter_pos,20,20, int(20+5*cnn_layer//2) ,upscaling_steps=upscaling_steps, blur=blur)
        # Layer visualization with pytorch hooks
        [img0,img1] = layer_vis.visualise_layer_with_hooks()
        axes[0,i].imshow(img0,cmap='gray')
        axes[1,i].imshow(img1,cmap='gray')
        axes[2,i].imshow((img0-img1).mean(axis=2),cmap='RdBu')

    axes[0,0].set_ylabel('im0',rotation=0.,labelpad=20)
    axes[1,0].set_ylabel('im1',rotation=0.,labelpad=20)
    axes[2,0].set_ylabel('im0-im1',rotation=0.,labelpad=30)

    for i, ax in enumerate(axes.flatten()):
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle('Layer %i, upscaling steps=%i, blur=%i, filter_sjze=%i,recep_size=%i' %(cnn_layer,upscaling_steps,blur,img1.shape[0],recep_field_size))
    save= True
    if save:
        fig.savefig('./plots/layer%i-actviz.pdf'%(cnn_layer))
    plt.show()

    # im0 = imread('./frame1.png').astype(np.float32)
    # im1 = imread('./frame2.png').astype(np.float32)
    # im0_mean = np.array((0.45014125, 0.4320596, 0.4114511))
    # im1_mean = np.array((0.44855332, 0.43102142,0.41060242))
    #
    # # switch between these 2
    # im0_norm = (im0/255.)-im0_mean
    # im1_norm = (im1/255.)-im1_mean
    # images = np.concatenate((im0_norm,im1_norm),axis=2).transpose(2,0,1).reshape(1,6,384,512)
    #
    # # images = np.array((im0, im1)).transpose(3,0,1,2).reshape(1, 3, 2, 384, 512)
    # images = torch.from_numpy(images.astype(np.float32)).cuda()
    # x = images
    # for i,layer in enumerate(all_layers):
    #     x = layer(x)
    #     if i in [1,5,9,13,19]:
    #         plt.plot(np.arange(x.shape[1]),np.mean(x.data.cpu().numpy()[0,:,:,:],axis=(1,2)))
    #         plt.title('activation layer %i' %(i))
    #         plt.show()

    # with torch.no_grad():
    #     output = model_and_loss.model(images)
    #     # output = model_and_loss.model(images.view(1,3,2,384,512))
    # flow = output[0].data.cpu().numpy().transpose(1, 2, 0)
    # plt.imshow(mmcv.flow2rgb(flow))
    # plt.show()



