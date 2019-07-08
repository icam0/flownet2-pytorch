import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('./temp/aperture_mag_bg/inf_stats.pickle','rb') as f:
    stats_ap = pickle.load(f)
with open('./temp/motion_mag_bg/inf_stats.pickle','rb') as f:
    stats_motion = pickle.load(f)
with open('./temp/motion_mag_bg2/inf_stats.pickle','rb') as f:
    stats_motion2 = pickle.load(f)
with open('./temp/motion_mag_bg2/inf_stats_C.pickle','rb') as f:
    stats_motion2C = pickle.load(f)

with open('./temp/aperture_scale_bg/inf_stats.pickle','rb') as f:
    stats_scale = pickle.load(f)
with open('./temp/occlusion_bg/inf_stats.pickle','rb') as f:
    stats_occ = pickle.load(f)
with open('./temp/orientation_bg/inf_stats.pickle','rb') as f:
    stats_orien = pickle.load(f)

width = 512
height = 384
square_size = 64

#u magnitude range
sx_range = np.arange(10,450,40).tolist()
sx_range = [2] + sx_range + [width-square_size]

#motion mag2 range
sx_range2 = np.arange(2,220,2)

#u+v magnitude range
s_range = np.arange(10,330,40).tolist()
s_range = [2] + s_range +[height-square_size]
s_mag = [(2*s**2)**0.5 for s in s_range]

#scale range
#sx,sy = 50,50
square_size_range = np.arange(8,334,8).tolist() + [334]

#occ_range
occ_range = np.arange(0,68,4)

#orientation range
orien_range = np.arange(0,360,5)

#convert to arrays
stats_ap = np.array(stats_ap)
stats_motion = np.array(stats_motion)
stats_motion2= np.array(stats_motion2)
stats_motion2C= np.array(stats_motion2C)
stats_scale = np.array(stats_scale)
stats_occ = np.array(stats_occ)
stats_orien = np.array(stats_orien)

plt.plot(sx_range,stats_motion[:,1],label='u')
plt.plot(s_mag,stats_ap[:,1],label='u+v')
plt.legend()
plt.ylabel('End-Point Error [-]')
plt.xlabel('Magnitude of motion [pixels/frame]')

plt.figure()
plt.plot(square_size_range,np.sqrt(stats_scale[:,1])) #remember the square root transform for the quadratic surface increase

plt.figure()
plt.plot(occ_range,stats_occ[:,1])
plt.xlabel('Occlusion width in pixels')
plt.ylabel('End-Point Error [-')
plt.show()

plt.figure()
plt.plot(orien_range,stats_orien[:,1])
plt.xlabel('Orientation angle in degrees')
plt.ylabel('End-Point Error [-')
plt.show()

plt.figure()
plt.plot(sx_range2,stats_motion2[:,1],label='S')
plt.plot(sx_range2,stats_motion2C[:,1],label='C')

plt.ylabel('End-Point Error [-]')
plt.xlabel('Magnitude of motion [pixels/frame]')
plt.legend()

plt.show()
