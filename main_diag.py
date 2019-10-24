import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import mmcv
from scipy.misc import imsave,imread

from utils import tools, load_model, get_layers,get_crop,HookClass, load_dataset, flownet2def
from viz_toolkit.layer_activation_with_guided_backprop import GuidedBackprop
from viz_toolkit import misc_functions
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import pickle

#load the model and dataset
model,layers_flat = load_model.load_model()

#load image
im0 = imread('./input_images/frame1.png').astype(np.float32)
im1 = im0
#im1 = imread('./input_images/frame2.png').astype(np.float32)
im0_mean = np.array((0.45014125, 0.4320596, 0.4114511))
im1_mean = np.array((0.44855332, 0.43102142,0.41060242))

gt = mmcv.flowread('./input_images/gt.flo')

# switch between these 2
im0_norm = (im0/255.)-im0_mean
im1_norm = (im1/255.)-im1_mean
images = np.concatenate((im0_norm,im1_norm),axis=2).transpose(2,0,1).reshape(1,6,384,512)
images = torch.tensor(images.astype(np.float32), requires_grad=True,device='cuda')

visualisation = {}
def hook_fn(m, i, o):
    visualisation[m] = o
def get_all_layers(net):
    for name, layer in net._modules.items():
        if isinstance(layer, nn.Sequential):
            get_all_layers(layer)
        else:
            layer.register_forward_hook(hook_fn)
get_all_layers(model)
out= model(images)
visualisation.keys()

layer_flat_indices = [1,3,5,7,9,11,13,15,17,19]
max_per_layer = 4
test_name = 'no_vel'
if not os.path.exists('./viz_im/%s'%(test_name)):
    os.makedirs('./viz_im/%s'%(test_name))



with open('./act_diff/%s.pkl'%(test_name),'wb') as f:
    pickle.dump(visualisation,f)

# GBP = GuidedBackprop(layers_flat, mode='gb')
# fig,axes = plt.subplots(len(layer_flat_indices),1,figsize=(8.,12.),sharey=True)
#
# for i,layer_flat_idx in enumerate(layer_flat_indices):
#     print(layer_flat_idx)
#
#     #get the layer definitions and activations
#     layer_name_idx = (layer_flat_idx-1)//2
#     _, layer_name, fmap_h, fmap_w, filters_layer, recep_field_layer, reduction_power = flownet2def.get_layer_details(layer_name_idx, images.shape[2], images.shape[3])
#     size = int((recep_field_layer + 1) * 1.5)
#     if size > images.shape[2]:
#         size = images.shape[2]
#     if size > images.shape[3]:
#         size = images.shape[3]
#
#     #get the activations per layer
#     act = visualisation[list(visualisation.keys())[layer_flat_idx]].cpu().data.numpy()[0]
#
#     #define the numpy arrays
#     all_images = np.zeros((size, 2 * max_per_layer * size, 3))
#     all_grads = np.zeros((size,2*max_per_layer*size,3))
#     all_flow = np.zeros((size,2*max_per_layer*size,2))
#     max_ind,max_values = [],[]
#
#     #plot the maximum activations per filter
#     act_max = act.copy()
#     act_max = act.reshape(act.shape[0],-1)
#     act_max = np.amax(act_max, axis=1)
#     axes[i].plot(np.arange(filters_layer),act_max)
#
#     for j in range(max_per_layer):
#         ind = np.unravel_index(np.argmax(act, axis=None), act.shape)
#         max_ind.append(ind)
#         max_values.append(act[ind])
#         act[ind[0],:,:] = 0. #set the whole filter to zero
#
#     # plot the activations of the top x responses
#     for index, max_act in enumerate(max_ind):
#         # extract the original xypos
#         hw_pos = list(max_act[1:])
#
#         # load the original gt + input
#         grads = GBP.generate_gradients(images, layer_flat_idx, max_act[0], hw_pos)
#         flow_pred = model(images)
#         GBP.forward_relu_outputs = [] #clear the forward hooks after inference on image is done
#
#         crop_h,crop_w = get_crop.get_crop(hw_pos,reduction_power,size,images.shape[2], images.shape[3])
#         grad_im0 = grads[0:3,slice(*crop_h) , slice(*crop_w)].transpose(1, 2, 0)
#         grad_im1 = grads[3:, slice(*crop_h) , slice(*crop_w)].transpose(1, 2, 0)
#         all_grads[:,2*size*index  :   2*size*(index+1),:] = np.concatenate((grad_im0,grad_im1),axis=1)
#
#         im0 = images[0,0:3,slice(*crop_h) , slice(*crop_w)].data.cpu().numpy().transpose(1,2,0)
#         im1 = images[0, 3:, slice(*crop_h) , slice(*crop_w)].data.cpu().numpy().transpose(1,2,0)
#         im0,im1 = misc_functions.denormalize(im0,im1)
#         all_images[:, 2 * size * index:2 * size * (index + 1), :] = np.concatenate(
#             (im0, im1), axis=1)
#
#         gt_temp = gt[slice(*crop_h) , slice(*crop_w),:]
#         flow_pred = flow_pred[0,:,slice(*crop_h) , slice(*crop_w)].data.cpu().numpy().transpose(1,2,0)
#         all_flow[:,2* size * index: 2*size*(index+1), :] = np.concatenate((gt_temp,flow_pred),axis=1)
#     # Normalize grads
#     all_grads = all_grads - all_grads.min()
#     all_grads /= all_grads.max()
#     tot_im = np.concatenate((all_grads,all_images,mmcv.flow2rgb(all_flow)),axis=0)
#     imsave('./viz_im/%s/%s.png'%(test_name,layer_name) ,tot_im)
#     np.savetxt('./viz_im/%s/%s.txt'%(test_name,layer_name),np.concatenate((np.array(max_ind),np.array(max_values)[:,np.newaxis]),axis=1),fmt='%5.5f')
# fig.savefig('./viz_im/%s/max_act.pdf'%(test_name))