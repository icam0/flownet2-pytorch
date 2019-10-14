import torch
from utils import tools
import datasets
import argparse

def load_dataset():
    # Parse arguments
    parser = argparse.ArgumentParser()
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

    args.inference_dataset_class = tools.module_to_dict(datasets)[args.inference_dataset]
    args.effective_number_workers = args.number_workers * args.number_gpus
    gpuargs = {'num_workers': args.effective_number_workers,
               'pin_memory': True,
               'drop_last': True} if args.cuda else {}
    inf_gpuargs = gpuargs.copy()
    inf_gpuargs['num_workers'] = args.number_workers

    # load data
    inference_dataset = args.inference_dataset_class(args, False, **tools.kwargs_from_args(args, 'inference_dataset'))
    inference_loader = torch.utils.data.DataLoader(inference_dataset, batch_size=1, shuffle=False, **inf_gpuargs)

    return inference_dataset, inference_loader

