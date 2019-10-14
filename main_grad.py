import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import torch
import numpy as np
import mmcv
from scipy.misc import imsave

from utils import tools, load_model, get_layers,get_crop,HookClass, load_dataset, flownet2def
from viz_toolkit.layer_activation_with_guided_backprop import GuidedBackprop
from viz_toolkit import misc_functions
from tqdm import tqdm
import pandas as pd

#load the model and dataset
model,layers_flat = load_model.load_model()
inference_dataset, inference_loader = load_dataset.load_dataset()
inp_h, inp_w = inference_dataset.render_size[0], inference_dataset.render_size[1]

#get the layer details and register forward hook
layer_name_idx = 4
layer_flat_idx = layer_name_idx*2+1
nr_of_samples, layer_name, fmap_h, fmap_w, filters_layer, recep_field_layer, reduction_power = flownet2def.get_layer_details(layer_name_idx,inp_h,inp_w)
hook_class = HookClass.HookClass(model,layer_name,nr_of_samples,filters_layer,fmap_h,fmap_w)

#perform inference
progress = tqdm(inference_loader, desc='Recording activations layer %s' % (layer_name), total=nr_of_samples - 1)
for index, (data, target) in enumerate(progress):
    if index == nr_of_samples:
        progress.close()
        break
    data, target = [d.cuda(non_blocking=True) for d in data], [t.cuda(non_blocking=True) for t in target]
    data[0] = torch.cat((data[0][:, :, 0, :, :], data[0][:, :, 1, :, :]), dim=1).requires_grad_()
    with torch.no_grad():
        hook_class.model(data[0])

#copy the data to the cpu and remove hook
layer_act = hook_class.activations[:,:,:,:].cpu().data.numpy()
del hook_class.activations
hook_class.handle.remove()

#find top x filters based on average top x actviations
amount_filters = filters_layer
max_filters_ind = range(amount_filters)

max_amount = 20 #nr of samples
grad_df = pd.DataFrame(index=range(max_amount*amount_filters) ,columns=['filt_nr','img0_grad','img1_grad','diff_grad'])
# Get gradients
GBP = GuidedBackprop(layers_flat, mode='backprop')
for c,filt_nr in enumerate(tqdm(max_filters_ind,desc='Guided backprop')):

    max_indices = []
    size = int((recep_field_layer+1))#*1.5

    #find the locations
    for i in range(max_amount):
        ind = np.unravel_index(np.argmax( layer_act[:,filt_nr,:,:], axis=None), layer_act[:,filt_nr,:,:].shape)
        max_indices.append(ind)
        layer_act[ind[0],filt_nr,ind[1],ind[2]] = 0. #set the previous value to zero to find the new maximum

    #plot the activations of the top x responses
    for index,max_act in enumerate(max_indices):
        #extract the original xypos
        hw_pos = list(max_act[1:])

        #load the original gt + input
        images = inference_dataset[max_act[0]][0][0].view(-1,3,2,384,1024).cuda()
        images = torch.cat((images[:, :, 0, :, :], images[:, :, 1, :, :]), dim=1).requires_grad_()
        grads = GBP.generate_gradients(images, layer_flat_idx, filt_nr,hw_pos)

        # Get the crop indices
        crop_h,crop_w = get_crop.get_crop(hw_pos, reduction_power, size,inp_h,inp_w)
        grad_im0 = grads[0:3,slice(*crop_h) , slice(*crop_w)].transpose(1, 2, 0)
        grad_im1 = grads[3:, slice(*crop_h) , slice(*crop_w)].transpose(1, 2, 0)
        grad_im0_m = np.abs(grad_im0).mean()
        grad_im1_m = np.abs(grad_im1).mean()
        grad_df.loc[c*max_amount+index] = [filt_nr,grad_im0_m,grad_im1_m,grad_im0_m-grad_im1_m]
save_pkl = True
if save_pkl:
    grad_df.to_pickle('./grads/%s-backprop-%i.pkl'%(layer_name,max_amount))

