import torch
import torch.nn as nn

from utils import flow_utils, tools
import models, losses, datasets
import argparse, os, sys, subprocess
from scipy.misc import imread
import numpy as np

with torch.cuda.device(0):
    class ModelAndLoss(nn.Module):
        def __init__(self, args):
            super(ModelAndLoss, self).__init__()
            kwargs = tools.kwargs_from_args(args, 'model')
            self.model = args.model_class(args, **kwargs)
            kwargs = tools.kwargs_from_args(args, 'loss')
            self.loss = args.loss_class(args, **kwargs)

        def forward(self, data, target, inference=False ):
            output = self.model(data)

            loss_values = self.loss(output, target)

            if not inference :
                return loss_values
            else :
                return loss_values, output

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', default='./checkpoints/FlowNet2-S_checkpoint.pth.tar', type=str, metavar='PATH')
    tools.add_arguments_for_module(parser, losses, argument_for_class='loss', default='L2Loss')
    tools.add_arguments_for_module(parser, models, argument_for_class='model', default='FlowNet2S')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument("--rgb_max", type=float, default = 255.)
    args = parser.parse_args()
    args.model_class = tools.module_to_dict(models)[args.model]
    args.loss_class = tools.module_to_dict(losses)[args.loss]

    # Load the model and loss
    model_and_loss = ModelAndLoss(args)
    model_and_loss = model_and_loss.cuda()
    # model_and_loss = nn.parallel.DataParallel(model_and_loss, device_ids=list(range(args.number_gpus)))
    torch.cuda.manual_seed(args.seed)

    # Load previous checkpoint
    checkpoint = torch.load(args.resume)
    model_and_loss.model.load_state_dict(checkpoint['state_dict'])
    model_and_loss.eval()

    img1 = torch.from_numpy(imread('./frame1.png').astype(np.float32))
    img2 = torch.from_numpy(imread('./frame2.png').astype(np.float32))
    img_tensor = torch.cat((img1, img2), 2).cuda(non_blocking=True)
    gt = torch.from_numpy(flow_utils.readFlow('./frame1.flo').astype(np.float32)).cuda(non_blocking=True)

    with torch.no_grad():
        losses, output = model_and_loss(img_tensor, gt, inference=True)