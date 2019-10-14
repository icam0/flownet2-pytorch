import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import torch
import numpy as np
import mmcv
from scipy.misc import imsave
from skimage.transform import rescale

from utils import tools, load_model,get_crop,HookClass, load_dataset, flownet2def
from viz_toolkit.layer_activation_with_guided_backprop import GuidedBackprop
from viz_toolkit import misc_functions
from tqdm import tqdm

#load the model and dataset
model,layers_flat = load_model.load_model()
inference_dataset, inference_loader = load_dataset.load_dataset()
inp_h, inp_w = inference_dataset.render_size[0], inference_dataset.render_size[1]

#get the layer details and register forward hook
layer_name_idx = 9
layer_flat_idx = layer_name_idx*2+1
nr_of_samples, layer_name, fmap_h, fmap_w, filters_layer, recep_field_layer, reduction_power = flownet2def.get_layer_details(layer_name_idx,inp_h,inp_w)
hook_class = HookClass.HookClass(model,layer_name,nr_of_samples,filters_layer,fmap_h,fmap_w)

get_samp = 20
nr_of_samples = get_samp+1
#perform inference
progress = tqdm(inference_loader, desc='Recording activations layer %s' % (layer_name), total=nr_of_samples - 1)
for index, (data, target) in enumerate(progress):
    if index == nr_of_samples:
        progress.close()
        break
    data, target = [d.cuda(non_blocking=True) for d in data], [t.cuda(non_blocking=True) for t in target]
    data[0] = torch.cat((data[0][:, :, 0, :, :], data[0][:, :, 1, :, :]), dim=1).requires_grad_()
    with torch.no_grad():
        if index == get_samp:
            hook_class.model.training = True
            flows = hook_class.model(data[0])
            im0 = data[0][0,0:3,: , :].data.cpu().numpy().transpose(1,2,0)
            im1 = data[0][0, 3:, : , :].data.cpu().numpy().transpose(1,2,0)
            gt = target[0][0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)
            im0,im1 = misc_functions.denormalize(im0,im1)
            hook_class.model.training = False
        else:
            hook_class.model(data[0])

#copy the data to the cpu and remove hook
layer_act = hook_class.activations[:,:,:,:].cpu().data.numpy()
del hook_class.activations
hook_class.handle.remove()

for i,flow in enumerate(flows[::-1]): #start from flow6
    flow = flow[0].data.cpu().numpy().transpose(1,2,0)
    flow_up = rescale(flow, scale=256./flow.shape[1], multichannel=True, order=0, anti_aliasing=False)
    if i== 0:
        tot_im = flow_up
    else:
        tot_im = np.concatenate((tot_im,flow_up),axis=1)
tot_im = mmcv.flow2rgb(tot_im)
imsave('./inter_flow/gt.png',mmcv.flow2rgb(gt))
imsave('./inter_flow/inter_flow_mpi.png',tot_im)
imsave('./inter_flow/mpi_input0.png',im0)
imsave('./inter_flow/mpi_input1.png',im1)

plt.imshow(tot_im)
plt.show()

#find top x filters based on average top x actviations
# print('finding maximum filters')
# amount_filters = 40
# max_filters = np.copy(layer_act)
# max_filters = max_filters.transpose(1,0,2,3)
# max_filters = max_filters.reshape(max_filters.shape[0],-1)
# mean_layer_act = np.partition(max_filters, -amount_filters,axis=1)[:,-amount_filters:].mean(axis=1)
# max_filters_ind = np.argpartition(mean_layer_act,-amount_filters)[-amount_filters:]
#
# #perform inference
# GBP = GuidedBackprop(layers_flat, mode='gb')
# max_amount = 16 #nr of samples
# for c,filt_nr in enumerate(tqdm(max_filters_ind,desc='Guided backprop')):
#     max_indices = []
#     size = int((recep_field_layer+1))#*1.5
#     all_images = np.zeros((size, 2 * max_amount * size, 3))
#     all_grads = np.zeros((size,2*max_amount*size,3))
#     all_flow = np.zeros((size,2*max_amount*size,2))
#
#     #find the locations
#     for i in range(max_amount):
#         ind = np.unravel_index(np.argmax( layer_act[:,filt_nr,:,:], axis=None), layer_act[:,filt_nr,:,:].shape)
#         max_indices.append(ind)
#         layer_act[ind[0],filt_nr,ind[1],ind[2]] = 0. #set the previous value to zero to find the new maximum
#
#     #plot the activations of the top 8 responses
#     for index,max_act in enumerate(max_indices):
#         #extract the original xypos
#         hw_pos = list(max_act[1:])
#
#         #load the original gt + input
#         gt = inference_dataset[max_act[0]][1][0]
#         images = inference_dataset[max_act[0]][0][0].view(-1,3,2,384,1024).cuda()
#         images = torch.cat((images[:, :, 0, :, :], images[:, :, 1, :, :]), dim=1).requires_grad_()
#
#         grads = GBP.generate_gradients(images, layer_flat_idx, filt_nr,hw_pos)
#         flow_pred = model(images)
#         GBP.forward_relu_outputs = [] #clear the forward hooks after inference on image is done
#
#         # Get the crop indices
#         crop_h,crop_w = get_crop.get_crop(hw_pos,reduction_power,size,inp_h,inp_w)
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
#         gt = gt[:,slice(*crop_h) , slice(*crop_w)].data.cpu().numpy().transpose(1,2,0)
#         flow_pred = flow_pred[0,:,slice(*crop_h) , slice(*crop_w)].data.cpu().numpy().transpose(1,2,0)
#         all_flow[:,2* size * index: 2*size*(index+1), :] = np.concatenate((gt,flow_pred),axis=1)
#
#     # Normalize grads
#     all_grads = all_grads - all_grads.min()
#     all_grads /= all_grads.max()
#     imsave('./viz_flownet/large/%s_filter%i.png'%(layer_name,filt_nr) ,np.concatenate((all_grads,all_images,mmcv.flow2rgb(all_flow)),axis=0))