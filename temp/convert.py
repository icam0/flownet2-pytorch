import imageio
import mmcv
import numpy as np
import torch
import matplotlib.pyplot as plt
import subprocess
import pickle

# sx_range = np.arange(2,220,2)
#
# for sx in sx_range:
#     frame1 = './motion_mag_bg2/clean/s_%03i/frame_0001.ppm' %(sx)
#     frame2 = './motion_mag_bg2/clean/s_%03i/frame_0002.ppm' %(sx)
#     subprocess.run("./ldof %s %s" %(frame1,frame2),shell=True)
#     print('done sx: %i' %(sx))

#
# losses = []
# for sx in sx_range:
#     gt = mmcv.flowread('./motion_mag_bg2/flow/s_%03i/frame_0001.flo' %(sx))
#     pred_ldof = mmcv.flowread('./motion_mag_bg2/clean/s_%03i/frame_0001LDOF.flo'%(sx))
#     #pred_s = mmcv.flowread('motion_mag_bg2/pred/000010.flo')
#     # plt.imshow(mmcv.flow2rgb(gt))
#     # plt.show()
#     # plt.imshow(mmcv.flow2rgb(pred_ldof))
#     # plt.show()
#     # plt.imshow(mmcv.flow2rgb(pred_s))
#     # plt.show()
#
#     losses.append(np.linalg.norm(pred_ldof-gt,axis=2).mean())
#
# with open('./motion_mag_bg2/inf_stats_ldof.pickle','wb') as f:
#     pickle.dump(losses,f,protocol=pickle.HIGHEST_PROTOCOL)


flow_img = mmcv.flow2rgb(mmcv.flowread('./motion_mag_bg2/clean/s_100/frame_0001LDOF.flo'))
imageio.imsave('./ldof_fail.png',flow_img)



