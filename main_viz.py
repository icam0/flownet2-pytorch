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
from viz_toolkit import misc_functions
import warnings

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

#load data
inference_dataset = args.inference_dataset_class(args, False, **tools.kwargs_from_args(args, 'inference_dataset'))
inference_loader = torch.utils.data.DataLoader(inference_dataset, batch_size=1, shuffle=False, **inf_gpuargs)
nr_of_samples = 1041
inp_h,inp_w = 384,1024
filters_nr = [64,128,256,256,512,512,512,512,1024,1024,1024]
recep_field = [7, 15, 31, 47, 63, 95, 127, 191, 255, 383,511]
stride_power = [1,2,3,3,4,4,5,5,6,6,6]
layer_names = ['conv1','conv2','conv3','conv3_1','conv4','conv4_1','conv5','conv5_1','conv6','conv6_1','predict_flow6']
cnn_layer_idx = 9

layer_name = layer_names[cnn_layer_idx]
recep_field_layer = recep_field[cnn_layer_idx]
fmap_h,fmap_w = int(inference_dataset[0][0][0].shape[2]/(2**(stride_power[cnn_layer_idx]))), int(inference_dataset[0][0][0].shape[3]/(2**(stride_power[cnn_layer_idx])))
hook_class = HookClass.HookClass(model,layer_name,nr_of_samples,filters_nr[cnn_layer_idx],fmap_h,fmap_w)

print('inferencing on layer %s' %(layer_name))
for index,(data,target) in enumerate(inference_loader):
    if index == nr_of_samples:
        break
    if args.cuda:
        data, target = [d.cuda(non_blocking=True) for d in data], [t.cuda(non_blocking=True) for t in target]
        data[0] = torch.cat((data[0][:, :, 0, :, :], data[0][:, :, 1, :, :]), dim=1).requires_grad_()
        with torch.no_grad():
            hook_class.model(data[0])

#copy the data to the cpu
layer_act = hook_class.activations[:,:,:,:].cpu().data.numpy()
del hook_class.activations

#remove original forward hook
hook_class.handle.remove()

#find top x filters based on average top x actviations
print('finding maximum')
amount_filters = 8
max_filters = np.copy(layer_act)
max_filters = max_filters.transpose(1,0,2,3)
max_filters = max_filters.reshape(max_filters.shape[0],-1)
mean_layer_act = np.partition(max_filters, -amount_filters,axis=1)[:,-amount_filters:].mean(axis=1)
max_filters_ind = np.argpartition(mean_layer_act,-amount_filters)[-amount_filters:]

print('performing guided backprop')
#find locations of top x activations per filter and plot these
for c,filt_nr in enumerate(max_filters_ind):
    max_amount = 8
    max_indices = []
    size = int((recep_field_layer+1))#*1.5
    all_images = np.zeros((size, 2 * max_amount * size, 3))
    all_grads = np.zeros((size,2*max_amount*size,3))
    all_flow = np.zeros((size,2*max_amount*size,2))

    #find the locations
    for i in range(max_amount):
        ind = np.unravel_index(np.argmax( layer_act[:,filt_nr,:,:], axis=None), layer_act[:,filt_nr,:,:].shape)
        max_indices.append(ind)
        layer_act[ind[0],filt_nr,ind[1],ind[2]] = 0. #set the previous value to zero to find the new maximum

    #plot the activations of the top 8 responses
    for index,max_act in enumerate(max_indices):
        #extract the original xypos
        hw_pos = list(max_act[1:])

        #load the original gt + input
        gt = inference_dataset[max_act[0]][1][0]
        images = inference_dataset[max_act[0]][0][0].view(-1,3,2,384,1024).cuda()
        images = torch.cat((images[:, :, 0, :, :], images[:, :, 1, :, :]), dim=1).requires_grad_()

        # Get gradients
        GBP = GuidedBackprop(layers_flat,mode='gb',lr_relu=False)
        grads = GBP.generate_gradients(images, cnn_layer_idx*2, filt_nr,hw_pos)
        flow_pred = model(images)

        # Get the crop indices
        crop_h,crop_w = get_crop.get_crop(hw_pos, stride_power, cnn_layer_idx, size,inp_h,inp_w)
        grad_im0 = grads[0:3,slice(*crop_h) , slice(*crop_w)].transpose(1, 2, 0)
        grad_im1 = grads[3:, slice(*crop_h) , slice(*crop_w)].transpose(1, 2, 0)
        all_grads[:,2*size*index  :   2*size*(index+1),:] = np.concatenate((grad_im0,grad_im1),axis=1)

        im0 = images[0,0:3,slice(*crop_h) , slice(*crop_w)].data.cpu().numpy().transpose(1,2,0)
        im1 = images[0, 3:, slice(*crop_h) , slice(*crop_w)].data.cpu().numpy().transpose(1,2,0)
        im0,im1 = misc_functions.denormalize(im0,im1)
        all_images[:, 2 * size * index:2 * size * (index + 1), :] = np.concatenate(
            (im0, im1), axis=1)

        gt = gt[:,slice(*crop_h) , slice(*crop_w)].data.cpu().numpy().transpose(1,2,0)
        flow_pred = flow_pred[0,:,slice(*crop_h) , slice(*crop_w)].data.cpu().numpy().transpose(1,2,0)
        all_flow[:,2* size * index: 2*size*(index+1), :] = np.concatenate((gt,flow_pred),axis=1)

    # Normalize grads
    all_grads = all_grads - all_grads.min()
    all_grads /= all_grads.max()
    print('Saving filter %i/%i'%(c+1,max_amount))
    imsave('./viz_flownet/%s_filter%i-nolr.png'%(layer_name,filt_nr) ,np.concatenate((all_grads,all_images,mmcv.flow2rgb(all_flow)),axis=0))


