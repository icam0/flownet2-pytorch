import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import torch
import numpy as np
import mmcv
from scipy.misc import imsave
from skimage.transform import rescale
from cmath import rect, phase
from math import radians, degrees
import pandas as pd
import seaborn as sns
from scipy.stats import circstd
sns.set()

from utils import tools, load_model,get_crop,HookClass, load_dataset, flownet2def
from viz_toolkit.layer_activation_with_guided_backprop import GuidedBackprop
from viz_toolkit import misc_functions
from tqdm import tqdm

def mean_angle(deg):
    return degrees(phase(sum(rect(1, radians(d)) for d in deg)/len(deg)))

#load the model and dataset
model,layers_flat = load_model.load_model()
inference_dataset, inference_loader = load_dataset.load_dataset()
inp_h, inp_w = inference_dataset.render_size[0], inference_dataset.render_size[1]

#get the layer details and register forward hook
layer_name_idx = 9
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
print('finding maximum filters')
amount_filters = 100 #number of filters to 'collect'
max_amount = 10 #nr of samples to base average on
max_filters = np.copy(layer_act)
max_filters = max_filters.transpose(1,0,2,3)
max_filters = max_filters.reshape(max_filters.shape[0],-1)
mean_layer_act = np.partition(max_filters, -max_amount,axis=1)[:,-max_amount:].mean(axis=1)
max_filters_ind = np.argpartition(mean_layer_act,-amount_filters)[-amount_filters:]

#perform inference
mode = 'backprop-check'
GBP = GuidedBackprop(layers_flat, mode=mode)
res =pd.DataFrame(index=max_filters_ind,columns=['mean_act','gt_mag_mean','gt_mag_std','pred_mag_mean','pred_mag_std','gt_angle_mean','gt_angle_std','pred_angle_mean','pred_angle_std'])
avg_grads_mag_all = np.zeros((amount_filters,6))
avg_img_mag_all = np.zeros((amount_filters,6))
for c,filt_nr in enumerate(tqdm(max_filters_ind,desc='Mode:'+mode)):
    max_indices = []
    size = int((recep_field_layer+1))#*1.5
    all_images = np.zeros((size, 2 * max_amount * size, 3))
    all_grads = np.zeros((size,2*max_amount*size,3))
    all_flow = np.zeros((size,2*max_amount*size,2))
    all_flow_mask = np.zeros((size,2*max_amount*size,2))

    avg_grads_mag = np.zeros((max_amount,6))
    avg_img_mag = np.zeros((max_amount, 6))

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

        grads = GBP.generate_gradients(images, layer_flat_idx, filt_nr,hw_pos)
        flow_pred = model(images)
        GBP.forward_relu_outputs = [] #clear the forward hooks after inference on image is done

        # Get the crop indices
        crop_h,crop_w = get_crop.get_crop(hw_pos,reduction_power,size,inp_h,inp_w)
        grad_im0 = grads[0:3,slice(*crop_h) , slice(*crop_w)].transpose(1, 2, 0)
        grad_im1 = grads[3:, slice(*crop_h) , slice(*crop_w)].transpose(1, 2, 0)
        grads = np.concatenate((grad_im0,grad_im1),axis=1)

        all_grads[:,2*size*index  : 2*size*(index+1),:] = grads

        im0 = images[0,0:3,slice(*crop_h) , slice(*crop_w)].data.cpu().numpy().transpose(1,2,0)
        im1 = images[0, 3:, slice(*crop_h) , slice(*crop_w)].data.cpu().numpy().transpose(1,2,0)
        im0,im1 = misc_functions.denormalize(im0,im1)
        all_images[:, 2 * size * index:2 * size * (index + 1), :] = np.concatenate((im0, im1), axis=1)

        gt = gt[:,slice(*crop_h) , slice(*crop_w)].data.cpu().numpy().transpose(1,2,0)
        flow_pred = flow_pred[0,:,slice(*crop_h) , slice(*crop_w)].data.cpu().numpy().transpose(1,2,0)
        all_flow[:,2* size * index: 2*size*(index+1), :] = np.concatenate((gt,flow_pred),axis=1)

        #get the mask
        threshold = np.percentile(np.abs(grads.flatten()), 95)  # get the top x percent
        mask = np.abs(np.copy(grad_im0))
        mask[mask < threshold] = 0.
        mask[mask >= threshold] = 1.
        mask = np.sum(mask,axis = 2)
        mask[mask > 1.] = 1.
        mask = mask[:,:,np.newaxis]
        rgb_mask = np.concatenate((mask,mask,mask),axis=2)
        flow_mask = np.concatenate((mask,mask),axis=2)
        gt_mask = np.multiply(gt,flow_mask)
        flow_pred_mask = np.multiply(flow_pred,flow_mask)
        all_flow_mask[:, 2 * size * index: 2 * size * (index + 1), :] = np.concatenate((gt_mask, flow_pred_mask), axis=1)

        flow_pred_mask = flow_pred[flow_mask > 0.].reshape(-1,2)
        gt_pred_mask = gt[flow_mask > 0.].reshape(-1, 2)

        if index ==0:
            all_pred_mask = flow_pred_mask
            all_gt_mask = gt_pred_mask
        else:
            all_pred_mask = np.concatenate((all_pred_mask,flow_pred_mask),axis=0)
            all_gt_mask = np.concatenate((all_gt_mask, gt_pred_mask),axis=0)

        avg_grads_mag[index,0:3] = np.abs(grad_im0).reshape(-1,3).mean(axis=0)
        avg_grads_mag[index, 3:] = np.abs(grad_im1).reshape(-1, 3).mean(axis=0)

        avg_img_mag[index,0:3] = im0.reshape(-1,3).mean(axis=0)
        avg_img_mag[index, 3:] = im1.reshape(-1, 3).mean(axis=0)


    fig,axes = plt.subplots(2,1)
    pred_angle = np.arctan2(all_pred_mask[:, 1], all_pred_mask[:, 0])*180./np.pi
    pred_mag = np.sqrt(all_pred_mask[:,0]**2. + all_pred_mask[:,1]**2.)
    gt_angle = np.arctan2(all_gt_mask[:, 1], all_gt_mask[:, 0])*180./np.pi
    gt_mag = np.sqrt(all_gt_mask[:,0]**2. + all_gt_mask[:,1]**2.)
    
    gt_mag_mean = np.mean(gt_mag)
    gt_mag_std = np.std(gt_mag)
    pred_mag_mean = np.mean(pred_mag)
    pred_mag_std = np.std(pred_mag)
    gt_angle_mean = mean_angle(gt_angle)
    gt_angle_std = circstd(gt_angle,low=-180.,high=180.)
    pred_angle_mean = mean_angle(pred_angle)
    pred_angle_std = circstd(pred_angle,low=-180.,high=180.)

    res.loc[filt_nr] = [mean_layer_act[filt_nr],gt_mag_mean,gt_mag_std,pred_mag_mean,pred_mag_std,gt_angle_mean,gt_angle_std,pred_angle_mean,pred_angle_std]

    axes[0] = sns.distplot(gt_mag, ax=axes[0], label=r'gt $\mu=%.1f$ $\sigma=%.1f$'%(gt_mag_mean,gt_mag_std))
    axes[0] = sns.distplot(pred_mag,ax=axes[0],label=r'pred $\mu=%.1f$ $\sigma=%.1f$'%(pred_mag_mean,pred_mag_std))
    axes[1] = sns.distplot(gt_angle,ax=axes[1],label=r'gt $\mu=%.1f$ $\sigma=%.1f$'%(gt_angle_mean,gt_angle_std))
    axes[1] = sns.distplot(pred_angle, ax=axes[1], label=r'pred $\mu=%.1f$ $\sigma=%.1f$'%(pred_angle_mean,pred_angle_std))
    axes[1].set_xlabel('Orientation [degrees]')
    axes[0].set_xlabel('Flow magnitude [pixels]')
    axes[0].set_ylabel('Density')
    axes[1].set_ylabel('Density')
    axes[1].legend()
    axes[0].legend()
    axes[1].set_xlim([-180.,180.])
    axes[0].set_title(r'Flow distribution gradient mask top %i activations (filter %i)' % (max_amount,filt_nr))
    # fig.suptitle(r'Filter %i mean activation %.2f' % (filt_nr,mean_layer_act[filt_nr]))
    fig.tight_layout()
    # plt.savefig('./viz_flownet/%s_filter%i-bp-dist.pdf'%(layer_name,filt_nr))
    plt.close()

    # Normalize grads
    all_grads = all_grads - all_grads.min()
    all_grads /= all_grads.max()
    all_flow = np.concatenate((all_flow,all_flow_mask),axis=0)
    # imsave('./viz_flownet/mask/%s_filter%i-bp.png'%(layer_name,filt_nr) ,np.concatenate((all_grads,all_images,mmcv.flow2rgb(all_flow)),axis=0))

    avg_grads_mag_all[c,:] = np.mean(avg_grads_mag,axis=0)
    avg_img_mag_all[c, :] = np.mean(avg_img_mag, axis=0)

print(avg_grads_mag_all.mean(axis=0))
print(avg_img_mag_all.mean(axis=0))
# res.to_pickle('./viz_flownet/mask/overview-%i.pkl' %(amount_filters))

