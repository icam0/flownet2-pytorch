import torch

class HookClass():
    def __init__(self, model,layer_name,nr_of_samples,nr_of_filters,fmap_h,fmap_w):
        self.model = model
        self.activations = torch.zeros((nr_of_samples,nr_of_filters,fmap_h,fmap_w)).cuda()
        self.counter = 0
        self.layer_name = layer_name
        self.hook_layers()
    def hook_layers(self):
        def hook_fn(m, i, o):
            self.activations[self.counter,:,:,:] = (o[0,:,:,:])
            self.counter+=1
        for layer in self.model.named_modules():
            if layer[0] == self.layer_name+'.1':
                self.handle = layer[1].register_forward_hook(hook_fn)