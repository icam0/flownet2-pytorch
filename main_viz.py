import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import numpy as np
import mmcv

from utils import flow_utils, tools, gpu_selection, get_layers
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
filters_nr = [64,128,256,256,512,512,512,512,1024,1024,1024]
recep_field = [7, 15, 31, 47, 63, 95, 127, 191, 255, 383,511]
stride_power = [1,2,3,3,4,4,5,5,6,6,6]
layer_names = ['conv1','conv2','conv3','conv3_1','conv4','conv4_1','conv5','conv5_1','conv6','conv6_1','predict_flow6']
cnn_layer_idx = 6
layer_name = layer_names[cnn_layer_idx]
recep_field_layer = recep_field[cnn_layer_idx]

class HookClass():
    def __init__(self, model,layer_name,nr_of_samples,nr_of_filters,fmap_h,fmap_w):
        self.model = model
        self.activations = torch.zeros((nr_of_samples,nr_of_filters,fmap_h,fmap_w)).cuda()
        self.counter = 0
        self.layer_name = layer_name
        self.hook_layers()
    def hook_layers(self):
        def hook_fn(m, i, o):
            self.activations[self.counter,:,:,:] = (o[0,:,:,:])
            self.counter+=1
        for layer in self.model.named_modules():
            if layer[0] == self.layer_name+'.1':
                self.handle = layer[1].register_forward_hook(hook_fn)

fmap_h,fmap_w = int(inference_dataset[0][0][0].shape[2]/(2**(stride_power[cnn_layer_idx]))), int(inference_dataset[0][0][0].shape[3]/(2**(stride_power[cnn_layer_idx])))
hook_class = HookClass(model,layer_name,nr_of_samples,filters_nr[cnn_layer_idx],fmap_h,fmap_w)
for index,(data,target) in enumerate(inference_loader):
    print(index)
    if index == nr_of_samples:
        break
    if args.cuda:
        data, target = [d.cuda(non_blocking=True) for d in data], [t.cuda(non_blocking=True) for t in target]
        data[0] = torch.cat((data[0][:, :, 0, :, :], data[0][:, :, 1, :, :]), dim=1).requires_grad_()
        with torch.no_grad():
            hook_class.model(data[0])

#copy the data to the cpu
layer_act = hook_class.activations[:,:,:,:].cpu().data.numpy()

#remove original forward hook
hook_class.handle.remove()

#find top x filters based on average top x actviations
amount_filters = 8
max_filters = np.copy(layer_act)
max_filters = max_filters.transpose(1,0,2,3)
max_filters = max_filters.reshape(max_filters.shape[0],-1)
mean_layer_act = np.partition(max_filters, -amount_filters,axis=1)[:,-amount_filters:].mean(axis=1)
max_filters_ind = np.argpartition(mean_layer_act,-amount_filters)[-amount_filters:]

#find locations of top x activations per filter and plot these
for filnr in max_filters_ind:
    max_amount = 8
    max_indices = []

    #find the locations
    for i in range(max_amount):
        ind = np.unravel_index(np.argmax( layer_act[:,filnr,:,:], axis=None), layer_act[:,filnr,:,:].shape)
        max_indices.append(ind)
        layer_act[ind[0],filnr,ind[1],ind[2]] = 0.

    #plot the activations of the top 8 responses
    fig,axes = plt.subplots(4,max_amount,figsize=(12.,8.),)
    fig.suptitle('Layer %s, Filter %i' %(layer_names[cnn_layer_idx],filnr))
    for index,max_act in enumerate(max_indices):
        #extract the original xypos
        hw_pos = list(max_act[1:])

        #load the original images
        gt = inference_dataset[max_act[0]][1][0]
        images = inference_dataset[max_act[0]][0][0].view(-1,3,2,384,1024).cuda()
        images = torch.cat((images[:, :, 0, :, :], images[:, :, 1, :, :]), dim=1).requires_grad_()

        # Get gradients
        GBP = GuidedBackprop(layers_flat,mode=mode[0],lr_bprop=mode[1])
        guided_grads = GBP.generate_gradients(images, None, cnn_layer_idx*2, filnr,hw_pos)
        grad_mask = np.abs(guided_grads.copy())
        grad_mask[grad_mask > 0.03] = 1.
        grad_mask[grad_mask <=0.03] = 0.
        grad_mask = np.amax(grad_mask[0:3, :, :], axis=0)

        guided_grads = guided_grads - guided_grads.min()
        guided_grads /= guided_grads.max()

        crop_h = [int(hw_pos[0]*2**(stride_power[cnn_layer_idx])-0.75*recep_field_layer),int(hw_pos[0]*2**(stride_power[cnn_layer_idx])+0.75*recep_field_layer)]
        crop_w = [int(hw_pos[1] * 2 ** (stride_power[cnn_layer_idx]) - 0.75 * recep_field_layer),
                  int(hw_pos[1] * 2 ** (stride_power[cnn_layer_idx]) + 0.75 * recep_field_layer)]

        if crop_h[0] <0:
            crop_h[0] = 0
        if crop_h[1] > 384:
            crop_h[1] = 384
        if crop_w[0] <0:
            crop_w[0] = 0
        if crop_w[1] > 1024:
            crop_w[1] = 1024

        crop_h = tuple(crop_h)
        crop_w = tuple(crop_w)

        grad_img0 = guided_grads[0:3,slice(*crop_h) , slice(*crop_w)].transpose(1, 2, 0)
        grad_img1 = guided_grads[3:, slice(*crop_h) , slice(*crop_w)].transpose(1, 2, 0)
        grad_mask = grad_mask[slice(*crop_h), slice(*crop_w)]

        im0 = images[0,0:3,slice(*crop_h) , slice(*crop_w)].data.cpu().numpy().transpose(1,2,0)
        im1 = images[0, 3:, slice(*crop_h) , slice(*crop_w)].data.cpu().numpy().transpose(1,2,0)
        im0,im1 = misc_functions.denormalize(im0,im1)
        im0[im0<0.],im1[im1<0.] = 0.,0.
        gt_flow = gt[:,slice(*crop_h) , slice(*crop_w)].data.cpu().numpy().transpose(1,2,0)

        grads = np.concatenate((grad_img0,grad_img1),axis=1)
        imgs = np.concatenate((im0,im1),axis=1)
        gts = np.concatenate((mmcv.flow2rgb(gt_flow), mmcv.flow2rgb(gt_flow) * grad_mask[:, :, np.newaxis]), axis=1)
        tot_im = np.concatenate((grads,imgs),axis=0) #gts
        axes[index_mode,index].imshow(tot_im)
        # sns.distplot((gt_flow[:, :, 0]*grad_mask).flatten(), label='u',ax=axes[1,index])
        # sns.distplot((gt_flow[:, :, 1]*grad_mask).flatten(), label='v', ax=axes[1, index])
        # axes[1,index].legend()

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()






