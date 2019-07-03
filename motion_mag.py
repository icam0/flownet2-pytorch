import numpy as np
import matplotlib.pyplot as plt
import mmcv
from PIL import Image,ImageDraw
from skimage import io
import os

def save_img_pair(im1,im2,flow,mag,testset_name):
    mag = int(mag)
    flow_viz = mmcv.flow2rgb(flow)

    #make directories
    os.mkdir('./temp/%s/clean/s_%03i/'%(testset_name,mag))
    os.mkdir('./temp/%s/flow/s_%03i/'%(testset_name,mag))
    os.mkdir('./temp/%s/flow_viz/s_%03i/'%(testset_name,mag))

    #save image pair, gt and gt_viz
    im1.save('./temp/%s/clean/s_%03i/frame_0001.png' %(testset_name,mag))
    im2.save('./temp/%s/clean/s_%03i/frame_0002.png'%(testset_name,mag))
    mmcv.flowwrite(flow, './temp/%s/flow/s_%03i/frame_0001.flo'%(testset_name,mag))
    io.imsave('./temp/%s/flow_viz/s_%03i/flow_viz.png' %(testset_name,mag), flow_viz)

def generate_img_pair(widt,height,square_size,sx,sy,color,bg=None):
    if bg:
        im1 = bg.copy()
        im2 = bg.copy()
    else:
        im1 = Image.new('RGB', (width, height), color='white')
        im2 = Image.new('RGB', (width, height), color='white')

    d1 = ImageDraw.Draw(im1)
    d2 = ImageDraw.Draw(im2)

    #specify rectangle coordinates in the center of the screen
    x0 = width/2-0.5-(square_size/2 -1) #convert to pixel coordinates
    x1 = width/2+0.5+(square_size/2 -1) #convert to pixel coordinates
    y0 = height/2-0.5-(square_size/2 -1) #convert to pixel coordinates
    y1 = height/2+0.5+(square_size/2 -1)#convert to pixel coordinates

    #define start and end point
    d1.rectangle([(x0-0.5*sx, y0-0.5*sy), (x1-0.5*sx, y1-0.5*sy)],fill=color)
    d2.rectangle([(x0+.5*sx, y0+0.5*sy), (x1+0.5*sx, y1+0.5*sy)],fill=color)

    #convert to numpy array
    im1_np = np.array(im1)

    #generate ground truth
    flow = np.zeros((height,width,2))
    flow[im1_np[:,:,0] == 0,0] = sx
    flow[im1_np[:,:,0] == 0,1] = sy

    return im1,im2,flow

width = 512
height = 384
square_size = 64
#sx = 200. #make sure this is dividable by 2
sx_range = np.arange(10,450,40).tolist()
sx_range = [2] + sx_range + [width-square_size]
sy = 0.
color='black'
bg = Image.open("./temp/bg.png")
testset_name = 'motion_mag_bg'
for sx in sx_range:
    im1,im2,flow = generate_img_pair(width,height,square_size,sx,sy,color,bg)
    mag = (sx**2 + sy**2)**0.5
    save_img_pair(im1,im2,flow,mag,testset_name)