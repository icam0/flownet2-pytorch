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

class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model,layer_index, selected_filter,height,width,color=True,num_iter=20):
        self.model = model
        self.layer_index = layer_index
        self.selected_filter = selected_filter
        self.height = height
        self.width = width
        self.color=color
        self.num_iter = num_iter

        # Create the folder to export images if not exists
        if not os.path.exists('../generated'):
            os.makedirs('../generated')

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]
        # Hook the selected layer
        self.model[self.layer_index].register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self):
        # Hook the selected layer
        self.hook_layer()
        # Generate a random image
        if self.color:
            im0 = np.uint8(np.random.uniform(0., 255., (self.height, self.width, 3))) / 255.
            im1 = np.uint8(np.random.uniform(0., 255., (self.height, self.width, 3))) / 255.
            im0_norm,im1_norm = normalize(im0,im1)
        else:
            im0 = np.uint8(np.random.uniform(0., 255., (self.height, self.width))) / 255.
            im1 = np.uint8(np.random.uniform(0., 255., (self.height, self.width))) / 255.
            im0_norm = im0 - 0.43
            im1_norm = im1 - 0.43
            im0_norm = np.repeat(im0_norm[:, :, np.newaxis], 3, axis=2)
            im1_norm = np.repeat(im1_norm[:, :, np.newaxis], 3, axis=2)
        images = np.concatenate((im0_norm, im1_norm), axis=2).transpose(2, 0, 1).reshape(1, 6, self.height, self.width)
        # Process image and return variable
        # images = preprocess_image(random_image, False)
        # Define optimizer for the image
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
                if index == self.layer_index:
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
            # Clamp the values to keep them in original range
            if self.color:
                mean_values = np.array((0.45014125, 0.4320596, 0.4114511, 0.44855332, 0.43102142, 0.41060242))
                for i,mean in enumerate(mean_values):
                    images.data[0,i,:,:].clamp_(min=-mean,max=1.-mean)
            else:
                im0_gray = images.data[0,0:3,:,:].mean(dim=0)
                im1_gray = images.data[0,3:, :, :].mean(dim=0)
                images.data[0,0:3,:,:] = torch.cat((im0_gray.unsqueeze(0),im0_gray.unsqueeze(0),im0_gray.unsqueeze(0)),dim=0)
                images.data[0, 3:, :, :] = torch.cat((im1_gray.unsqueeze(0), im1_gray.unsqueeze(0), im1_gray.unsqueeze(0)),
                                                      dim=0)
                mean_value = 0.43
                images.data[0, :, :, :].clamp_(min=-mean_value, max=1. - mean_value)

        #return the separate images
        im0_norm,im1_norm = images.cpu().data.numpy()[0,0:3,:,:].transpose(1,2,0), images.cpu().data.numpy()[0,3:,:,:].transpose(1,2,0)
        #denormalize the images and return them
        if self.color:
            im0,im1 = denormalize(im0_norm,im1_norm)
        else:
            im0,im1 = im0_norm+0.43,im1_norm+0.43

        return im0, im1


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
