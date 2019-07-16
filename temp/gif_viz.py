import os
import imageio
from PIL import Image,ImageDraw,ImageFont
import numpy as np
png_dir = './motion_mag_bg2/pred/'
images = []
file_list = sorted(os.listdir(png_dir))
image_list = [image_path for image_path in file_list if image_path.endswith('.png')]
font = ImageFont.truetype('./arial.ttf',40)
sx_range = np.arange(2,220,2)
for sx,image_name in zip(sx_range,image_list):

    image_path = os.path.join(png_dir, image_name)
    #image = imageio.imread(file_path)
    image = Image.open(image_path)
    d = ImageDraw.Draw(image)
    d.text((192,20),'u=%i' %(int(sx)),fill='black',align='center',font=font)
    images.append(np.array(image))
imageio.mimsave('./movie.gif', images,format='GIF', duration=0.2)