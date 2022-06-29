import os
import torch
import fnmatch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import torch.utils.data.sampler as sampler
import shutil
from dataset.nyuv2ssl import *
from torch.autograd import Variable
from model.segnet_mtl import SegNet
from model.mapfns import Mapfns
from utils.evaluation import ConfMatrix, DepthMeter, NormalsMeter
import numpy as np
import pdb
from progress.bar import Bar as Bar
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
from torch.autograd import Variable
import copy

parser = argparse.ArgumentParser(description='Evaluation (SegNet)')
parser.add_argument('--type', default='standard', type=str, help='split type: standard, wide, deep')
parser.add_argument('--dataroot', default='./data/nyuv2', type=str, help='dataset root')
parser.add_argument('--model', default='', type=str, metavar='PATH', help='path to pre-trained model (default: none)')
parser.add_argument('--ssl-type', default='randomlabels', type=str, help='ssl type: onelabel, randomlabels, full')


opt = parser.parse_args()

tasks = ['semantic', 'depth', 'normal']
input_channels = [13, 1, 3]

stl_performance = {
                    'full': {'semantic': 37.447399999999995, 'depth': 0.607902, 'normal': 25.938105}, 
                    'onelabel': {'semantic': 26.1113, 'depth': 0.771502, 'normal': 30.073763}, 
                    'randomlabels': {'semantic': 28.7153, 'depth': 0.754012, 'normal': 28.946388}
}


# define model, optimiser and scheduler
model = SegNet(type_=opt.type, class_nb=13).cuda()
checkpoint = torch.load(opt.model)
model.load_state_dict(checkpoint['state_dict'], strict=False)


# compute parameter space
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print('Parameter Space: ABS: {:.1f}, REL: {:.4f}\n'.format(count_parameters(model),
                                                           count_parameters(model)/24981069))
print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR ROOT_MSE | NORMAL_LOSS MEAN MED <11.25 <22.5 <30 | Multi-task Performance\n')

# define dataset path
dataset_path = opt.dataroot

nyuv2_test_set = NYUv2(root=dataset_path, train=False)

batch_size = 2

nyuv2_test_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_test_set,
    batch_size=batch_size,
    shuffle=False, num_workers=0)


# define parameters
test_batch = len(nyuv2_test_loader)
avg_cost = np.zeros([12], dtype=np.float32)

model.eval()
conf_mat = ConfMatrix(model.class_nb)
depth_mat = DepthMeter()
normal_mat = NormalsMeter()
cost = np.zeros(12, dtype=np.float32)
with torch.no_grad():  # operations inside don't track history
    nyuv2_test_dataset = iter(nyuv2_test_loader)
    for k in range(test_batch):
        test_data, test_label, test_depth, test_normal = nyuv2_test_dataset.next()
        test_data, test_label = test_data.cuda(),  test_label.type(torch.LongTensor).cuda()
        test_depth, test_normal = test_depth.cuda(), test_normal.cuda()

        test_pred, _, _ = model(test_data)
        test_loss = model.model_fit(test_pred[0], test_label, test_pred[1], test_depth, test_pred[2], test_normal)

        conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())
        depth_mat.update(test_pred[1], test_depth)
        normal_mat.update(test_pred[2], test_normal)
        cost[0] = test_loss[0].item()
        cost[3] = test_loss[1].item()
        cost[6] = test_loss[2].item()

        avg_cost[0:] += cost[0:] / test_batch
    avg_cost[1:3] = conf_mat.get_metrics()
    depth_metric = depth_mat.get_score()
    avg_cost[4], avg_cost[5] = depth_metric['l1'], depth_metric['rmse']
    normal_metric = normal_mat.get_score()
    avg_cost[7], avg_cost[8], avg_cost[9], avg_cost[10], avg_cost[11] = normal_metric['mean'], normal_metric['rmse'], normal_metric['11.25'], normal_metric['22.5'], normal_metric['30']


mtl_performance = 0.0
mtl_performance += (avg_cost[1]* 100 - stl_performance[opt.ssl_type]['semantic']) / stl_performance[opt.ssl_type]['semantic']
mtl_performance -= (avg_cost[4] - stl_performance[opt.ssl_type]['depth']) / stl_performance[opt.ssl_type]['depth']
mtl_performance -= (avg_cost[7] - stl_performance[opt.ssl_type]['normal']) / stl_performance[opt.ssl_type]['normal']
mtl_performance = mtl_performance / len(tasks) * 100


print('TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} | {:.2f}'
        .format(avg_cost[0], avg_cost[1], avg_cost[2], avg_cost[3],
            avg_cost[4], avg_cost[5], avg_cost[6], avg_cost[7], avg_cost[8], avg_cost[9],
            avg_cost[10], avg_cost[11], mtl_performance))

