import warnings
import datasets
import losses as losses_flow
import models as flownet_models
import argparse, os, sys, subprocess
from utils import flow_utils, tools, gpu_selection, get_layers
import torch

def load_model():
    warnings.filterwarnings("ignore")

    # select the GPU
    free_gpu_id = gpu_selection.get_freer_gpu()
    print('Selecting GPU %i' % (free_gpu_id))
    torch.cuda.set_device(free_gpu_id)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', default='./checkpoints/FlowNet2-S_checkpoint.pth.tar', type=str, metavar='PATH')
    tools.add_arguments_for_module(parser, losses_flow, argument_for_class='loss', default='L2Loss')
    tools.add_arguments_for_module(parser, flownet_models, argument_for_class='model', default='FlowNet2S')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument("--rgb_max", type=float, default=255.)
    tools.add_arguments_for_module(parser, datasets, argument_for_class='inference_dataset', default='MpiSintelClean',
                                   skip_params=['is_cropped'],
                                   parameter_defaults={'root': '../MPI-Sintel/training',
                                                       'replicates': 1,
                                                       'norm_og': True})  # this activates the original normalization
    parser.add_argument('--crop_size', type=int, nargs='+', default=[256, 256],
                        help="Spatial dimension to crop training samples for training")
    parser.add_argument('--inference_size', type=int, nargs='+', default=[-1, -1],
                        help='spatial size divisible by 64. default (-1,-1) - largest possible valid size would be used')
    parser.add_argument('--number_workers', '-nw', '--num_workers', type=int, default=8)
    parser.add_argument('--number_gpus', '-ng', type=int, default=-1, help='number of GPUs to use')
    parser.add_argument('--inference_n_batches', type=int, default=-1)
    args = parser.parse_args()
    args.cuda = True
    args.model_class = tools.module_to_dict(flownet_models)[args.model]
    args.loss_class = tools.module_to_dict(losses_flow)[args.loss]
    args.inference_dataset_class = tools.module_to_dict(datasets)[args.inference_dataset]
    args.effective_number_workers = args.number_workers * args.number_gpus
    gpuargs = {'num_workers': args.effective_number_workers,
               'pin_memory': True,
               'drop_last': True} if args.cuda else {}
    inf_gpuargs = gpuargs.copy()
    inf_gpuargs['num_workers'] = args.number_workers

    # Load the model and loss
    kwargs = tools.kwargs_from_args(args, 'model')
    model = args.model_class(args, **kwargs)
    model = model.cuda()
    torch.cuda.manual_seed(args.seed)

    # Load previous checkpoint
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.zero_grad()

    layers_flat = get_layers.remove_sequential(model)
    layers_flat[20], layers_flat[28] = layers_flat[28], layers_flat[
        20]  # Swap the layers so we can directly loop until predict_flow 6
    layers_flat = layers_flat[0:21]
    return model,layers_flat