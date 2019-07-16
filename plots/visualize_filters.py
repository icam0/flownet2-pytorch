import torch
import matplotlib.pyplot as plt
import numpy as np
import skimage
wandb = torch.load('./checkpoint/FlowNet2-S_checkpoint.pth.tar')

conv1 = wandb['state_dict']['conv1.0.weight']

from matplotlib import pyplot as plt
import numpy as np

var_filter,var_filter_t0,var_filter_t1 = [],[],[]
for filter_idx in range(64):
    filter_t0 = conv1[filter_idx,0:3,:,:].numpy()
    filter_t1 = conv1[filter_idx,3:,:,:].numpy()

    var_t0 = np.mean(np.var(filter_t0,axis=0))
    var_t1 = np.mean(np.var(filter_t1,axis=0))

    mag_t0 = np.mean(np.abs(filter_t0))
    mag_t1 = np.mean(np.abs(filter_t1))

    var_filter_t0.append(var_t0 / mag_t0 * 100.)
    var_filter_t1.append(var_t1 / mag_t1 * 100.)

    variance_temporal = np.mean(np.var(filter_t0+filter_t1,axis=0))
    mag = np.mean(np.abs(filter_t0+filter_t1))
    var_filter.append(variance_temporal/mag *100.)

print(var_filter)
print(var_filter_t0)
print(var_filter_t1)

# filters = [9,14,16,44,60]
#
# fig,axes = plt.subplots(len(filters),3)
#
# for i,filter_idx in enumerate(filters):
#     filter = conv1[filter_idx,:,:,:]
#     filter_t0 = filter[0:3,:,:]
#     filter_t0_r = filter[0,:,:].numpy()
#     filter_t0_gray = filter_t0.mean(dim=0).numpy()
#     filter_t0_rgb = np.swapaxes(filter_t0.numpy(),0,2)
#
#     filter_t1 = filter[3:,:,:]
#     filter_t1_r = filter[3,:,:].numpy()
#     filter_t1_gray = filter_t1.mean(dim=0).numpy()
#     filter_t1_rgb = np.swapaxes(filter_t1.numpy(),0,2)
#
#     axes[i,0].imshow(skimage.transform.rescale(filter_t0_r,2, order=1), cmap='RdBu')
#     axes[i,1].imshow(skimage.transform.rescale(filter_t1_r,2,order=1),cmap='RdBu')
#     axes[i,2].imshow(skimage.transform.rescale(filter_t0_r,2,order=1)-skimage.transform.rescale(filter_t1_r,2,order=1),cmap='RdBu')
#
#     # axes[i,0].imshow(skimage.transform.rescale(filter_t0_gray,2, order=1),cmap='RdBu')
#     # axes[i,1].imshow(skimage.transform.rescale(filter_t1_gray,2,order=1),cmap='RdBu')
#     # axes[i,2].imshow(skimage.transform.rescale(filter_t0_gray,2,order=1)-skimage.transform.rescale(filter_t1_gray,2,order=1),cmap='RdBu')
#
# for i,ax in enumerate(axes.flatten()):
#     ax.set_xticks([])
#     ax.set_yticks([])
#     #ax.axis('off')
#     if i == 12:
#         ax.set_xlabel('t0')
#     if i == 13:
#         ax.set_xlabel('t1')
#     if i == 14:
#         ax.set_xlabel('t0-t1')
#
# fig.tight_layout(pad=0.05,w_pad=0.05,h_pad=1)
# plt.show()
#
#
#
