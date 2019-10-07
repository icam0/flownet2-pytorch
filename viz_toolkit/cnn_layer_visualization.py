"""
Created on Sat Nov 18 23:12:08 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import numpy as np

import torch
from torch.optim import Adam
from torchvision import models

from viz_toolkit.misc_functions import preprocess_image, recreate_image, save_image,normalize,denormalize
import matplotlib.pyplot as plt

import cv2

class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter,height,width,num_iter,upscaling_steps=1, blur = None,upscaling_factor=1.2):
        self.model = model
        # self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        self.height = height
        self.width = width
        self.num_iter = num_iter
        self.upscaling_steps = upscaling_steps
        self.blur = blur
        self.upscaling_factor = upscaling_factor

        # Create the folder to export images if not exists
        if not os.path.exists('../generated'):
            os.makedirs('../generated')

    def hook_layer(self):
        def hook_function_bw(module, grad_in, grad_out):
            print("------------Input Grad------------")

            for grad in grad_in:
                try:
                    print(grad.shape)
                except AttributeError:
                    print("None found for Gradient")

            print("------------Output Grad------------")
            for grad in grad_out:
                try:
                    print(grad.shape)
                except AttributeError:
                    print("None found for Gradient")
            print("\n")

        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]
        # Hook the selected layer
        self.model[self.selected_layer].register_forward_hook(hook_function)
        self.model[0].register_backward_hook(hook_function_bw)

    def visualise_layer_with_hooks(self):
        # Hook the selected layer
        self.hook_layer()
        # Generate a random image
        im0 = np.uint8(np.random.uniform(0., 255., (self.height, self.width, 3))) / 255.
        im1 = np.uint8(np.random.uniform(0., 255., (self.height, self.width, 3))) / 255.
        im0_norm,im1_norm = normalize(im0,im1)
        images = np.concatenate((im0_norm,im1_norm),axis=2).transpose(2,0,1).reshape(1,6,self.height,self.width)
        # Process image and return variable
        # images = preprocess_image(random_image, False)
        # Define optimizer for the image
        for _ in range(self.upscaling_steps):
            images = torch.tensor(images.astype(np.float32), requires_grad=True,device='cuda')
            optimizer = Adam([images], lr=0.1, weight_decay=1e-6)
            for i in range(1, self.num_iter):
                optimizer.zero_grad()
                # Assign create image to a variable to move forward in the model
                x = images
                for index, layer in enumerate(self.model):
                    # Forward pass layer by layer
                    # x is not used after this point because it is only needed to trigger
                    # the forward hook function
                    x = layer(x)
                    # Only need to forward until the selected layer is reached
                    if index == self.selected_layer:
                        # (forward hook function triggered)
                        break
                # Loss function is the mean of the output of the selected layer/filter
                # We try to minimize the mean of the output of that specific filter
                loss = -torch.mean(self.conv_output)
                print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.cpu().data.numpy()))
                # Backward
                loss.backward()
                # Update image
                optimizer.step()
                # for p in filter(lambda p: p.grad is not None, parameters):
                #     p.grad.data.clamp_(min=-clip_value, max=clip_value)
                mean_values = np.array((0.45014125, 0.4320596, 0.4114511, 0.44855332, 0.43102142, 0.41060242))
                for i,mean in enumerate(mean_values):
                    images.data[0,i,:,:].clamp_(min=-mean,max=1.-mean)
            #return the separate images
            im0_norm,im1_norm = images.cpu().data.numpy()[0,0:3,:,:].transpose(1,2,0), images.cpu().data.numpy()[0,3:,:,:].transpose(1,2,0)

            #upscale the image and blur it
            if self.upscaling_steps > 1:
                self.height,self.width = int(self.upscaling_factor * self.height), int(self.upscaling_factor * self.width)
                im0_norm = cv2.resize(im0_norm, (self.height,self.width ), interpolation=cv2.INTER_CUBIC)  # scale image up
                im1_norm = cv2.resize(im1_norm, (self.height,self.width ), interpolation=cv2.INTER_CUBIC)
                if self.blur is not None: im0_norm = cv2.blur(im0_norm, (self.blur, self.blur))  # blur image to reduce high frequency patterns
                if self.blur is not None: im1_norm = cv2.blur(im1_norm, (self.blur, self.blur))  # blur image to reduce high frequency patterns
                images = np.concatenate((im0_norm,im1_norm),axis=2).transpose(2,0,1).reshape(1,6,self.height,self.width)

        im0,im1 = denormalize(im0_norm,im1_norm)
        return im0, im1

    def visualise_layer_without_hooks(self):
        # Process image and return variable
        # Generate a random image
        random_image = np.uint8(np.random.uniform(150, 180, (self.height, self.width, 6)))
        # Process image and return variable
        images = preprocess_image(random_image, False)
        # Define optimizer for the image
        optimizer = Adam([images], lr=0.1, weight_decay=1e-6)
        for i in range(1, self.num_iter):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = images
            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                x = layer(x)
                if index == self.selected_layer:
                    # Only need to forward until the selected layer is reached
                    # Now, x is the output of the selected layer
                    break
            # Here, we get the specific filter from the output of the convolution operation
            # x is a tensor of shape 1x512x28x28.(For layer 17)
            # So there are 512 unique filter outputs
            # Following line selects a filter from 512 filters so self.conv_output will become
            # a tensor of shape 28x28
            self.conv_output = x[0, self.selected_filter]
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(images)
            # Save image
            if i % 5 == 0:
                im_path = '../generated/layer_vis_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                save_image(self.created_image, im_path)


if __name__ == '__main__':
    cnn_layer = 15
    filter_pos = 3
    # Fully connected layer is not needed
    pretrained_model = models.vgg16(pretrained=True).features
    layer_vis = CNNLayerVisualization(pretrained_model, cnn_layer, filter_pos)

    # Layer visualization with pytorch hooks
    layer_vis.visualise_layer_with_hooks()

    # Layer visualization without pytorch hooks
    # layer_vis.visualise_layer_without_hooks()
