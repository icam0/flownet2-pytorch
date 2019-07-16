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

    # im1.save('./temp/%s/clean/s_%03i/frame_0001.ppm' %(testset_name,mag))
    # im2.save('./temp/%s/clean/s_%03i/frame_0002.ppm'%(testset_name,mag))

    mmcv.flowwrite(flow, './temp/%s/flow/s_%03i/frame_0001.flo'%(testset_name,mag))
    io.imsave('./temp/%s/flow_viz/s_%03i/flow_viz.png' %(testset_name,mag), flow_viz)

def generate_img_pair(widt,height,square_size,sx,sy,color,bg=None,occlusion=None,angle=None,correspondence=False,color_2='black'):
    if bg:
        im1 = bg.copy()
        im2 = bg.copy()
    else:
        im1 = Image.new('RGB', (width, height), color='black')
        im2 = Image.new('RGB', (width, height), color='black')

    if angle:
        mag = sx
        sx = mag*np.cos(np.deg2rad(angle))
        sy = mag*np.sin(np.deg2rad(angle))

    d1 = ImageDraw.Draw(im1)
    d2 = ImageDraw.Draw(im2)

    #specify rectangle coordinates in the center of the screen
    x0 = width/2-0.5-(square_size/2 -1) #convert to pixel coordinates
    x1 = width/2+0.5+(square_size/2 -1) #convert to pixel coordinates
    y0 = height/2-0.5-(square_size/2 -1) #convert to pixel coordinates
    y1 = height/2+0.5+(square_size/2 -1)#convert to pixel coordinates

    #define start and end point
    if angle:
        d1.rectangle([(x0, y0), (x1, y1)],fill=color)
        d2.rectangle([(x0+sx, y0+sy), (x1+sx, y1+sy)],fill=color)

    elif correspondence:
        d1.rectangle([(x0-0.5*sx, y0-0.5*sy), (x1-0.5*sx, y1-0.5*sy)],fill=color)
        d1.rectangle([(x0 + .5 * sx, y0 + 0.5 * sy), (x1 + 0.5 * sx, y1 + 0.5 * sy)], fill=color_2)

        d2.rectangle([(x0+0.5*sx, y0-0.5*sy), (x1+0.5*sx, y1-0.5*sy)],fill=color)
        d2.rectangle([(x0 - .5 * sx, y0 + 0.5 * sy), (x1 - 0.5 * sx, y1 + 0.5 * sy)], fill=color_2)


    else:
        d1.rectangle([(x0-0.5*sx, y0-0.5*sy), (x1-0.5*sx, y1-0.5*sy)],fill=color)
        d2.rectangle([(x0+.5*sx, y0+0.5*sy), (x1+0.5*sx, y1+0.5*sy)],fill=color)



    # if occlusion or occlusion != 0:
    #     occlusion_diff = (square_size-occlusion)/2
    #     d1.rectangle([(x0+.5*sx+occlusion_diff, y0+0.5*sy-0.5*square_size), (x1+0.5*sx-occlusion_diff, y1+0.5*sy+0.5*square_size)],fill='white')
    #     d2.rectangle([(x0+.5*sx+occlusion_diff, y0+0.5*sy-0.5*square_size), (x1+0.5*sx-occlusion_diff, y1+0.5*sy+0.5*square_size)],fill='white')

    #convert to numpy array
    im1_np = np.array(im1)

    if color=='white':
        color_index = 255
    if color=='black':
        color_index= 0

    #generate ground truth
    flow = np.zeros((height,width,2))
    # flow[im1_np[:,:,0] == color_index,0] = sx
    # flow[im1_np[:,:,0] == color_index,1] = sy

    return im1,im2,flow

width = 512
height = 384
square_size = 32
color='black'
#sx = 200. #make sure this is dividable by 2

# sx_range = np.arange(1,450,40).tolist() #first sx_range
# sx_range = [2] + sx_range + [width-square_size]

sx_range = np.arange(2,220,2) #second sx_range

# s_range = np.arange(10,330,40).tolist()
# s_range = [2] + s_range +[height-square_size]
# sy = 0.
#square_size_range = np.arange(8,334,8).tolist() + [334]
#angle_range = np.arange(0,360,5)
# sx = 50
sx,sy = 64,64

bg = Image.open("./temp/bg.png")
#occlusion_range = np.arange(0,68,4)
testset_name = 'correspondence_bg'
occlusion = None
color_pairs = [('black','black'),('black','white'),('blue','red'),((200,0,0),(180,0,0))]

for i,color_pair in enumerate(color_pairs):
    im1,im2,flow = generate_img_pair(width,height,square_size,sx,sy,color=color_pair[0],bg=bg,occlusion=None,correspondence=True,color_2=color_pair[1])

    save_img_pair(im1,im2,flow,i,testset_name)