import numpy as np
import matplotlib.pyplot as plt
import mmcv
from PIL import Image,ImageDraw
from skimage import io
import os

def save_img_pair(im1,im2,flow,mag):
    mag = int(mag)
    flow_viz = mmcv.flow2rgb(flow)

    #make directories
    os.mkdir('./temp/motion_mag/clean/s_%i/'%(mag))
    os.mkdir('./temp/motion_mag/flow/s_%i/'%(mag))
    os.mkdir('./temp/motion_mag/flow_viz/s_%i/'%(mag))

    #save image pair, gt and gt_viz
    im1.save('./temp/motion_mag/clean/s_%i/frame_0001.png' %(mag))
    im2.save('./temp/motion_mag/clean/s_%i/frame_0002.png'%(mag))
    mmcv.flowwrite(flow, './temp/motion_mag/flow/s_%i/frame_0001.flo'%(mag))
    io.imsave('./temp/motion_mag/flow_viz/s_%i/flow_viz.png' %(mag), flow_viz)

def generate_img_pair(widt,height,square_size,sx,sy,color):

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

for sx in sx_range:
    im1,im2,flow = generate_img_pair(width,height,square_size,sx,sy,color)
    mag = (sx**2 + sy**2)**0.5
    save_img_pair(im1,im2,flow,mag)