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
from utils.evaluation import ConfMatrix, DepthMeter, NormalsMeter
import numpy as np
import pdb
from progress.bar import Bar as Bar
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='Single-task supervised learning (SegNet)')
parser.add_argument('--task', default='semantic', type=str, help='choose task: semantic, depth, normal')
parser.add_argument('--type', default='standard', type=str, help='split type: standard, wide, deep')
parser.add_argument('--weight', default='uniform', type=str, help='multi-task weighting: uniform')
parser.add_argument('--dataroot', default='./data/nyuv2', type=str, help='dataset root')
parser.add_argument('--out', default='./results/nyuv2', help='Directory to output the result')
parser.add_argument('--ssl-type', default='randomlabels', type=str, help='ssl type: onelabel, randomlabels, full')
parser.add_argument('--labelroot', default='./data/nyuv2_settings/', type=str, help='partially setting root')
parser.add_argument('--eval-last20', default=0, type=int, help='1 means we evaluate models in the last 20 epochs')

opt = parser.parse_args()

tasks = ['semantic', 'depth', 'normal']

if not os.path.isdir(opt.out):
    mkdir_p(opt.out)
paths = [opt.ssl_type, 'stl']
for i in range(len(paths)):
    opt.out = os.path.join(opt.out, paths[i])
    if not os.path.isdir(opt.out):
        mkdir_p(opt.out)

def save_checkpoint(state, is_best, checkpoint=opt.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, 'stl_sl_{}_{}_'.format(opt.ssl_type, opt.task) + filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'stl_sl_{}_{}_'.format(opt.ssl_type, opt.task) + 'model_best.pth.tar'))

title = 'NYUv2'
logger = Logger(os.path.join(opt.out, 'stl_sl_{}_{}_log.txt'.format(opt.ssl_type, opt.task)), title=title)
logger.set_names(['Epoch', 'T.Ls', 'T. mIoU', 'T. Pix', 'T.Ld', 'T.abs', 'T.rel', 'T.Ln', 'T.Mean', 'T.Med', 'T.11', 'T.22', 'T.30',
    'V.Ls', 'V. mIoU', 'V. Pix', 'V.Ld', 'V.abs', 'V.rel', 'V.Ln', 'V.Mean', 'V.Med', 'V.11', 'V.22', 'V.30', 'Ws', 'Wd', 'Wn'])

# define model, optimiser and scheduler
model = SegNet(type_=opt.type, class_nb=13).cuda()
params = []
params += model.parameters()
optimizer = optim.Adam(params, lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)


# compute parameter space
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print('Parameter Space: ABS: {:.1f}, REL: {:.4f}\n'.format(count_parameters(model),
                                                           count_parameters(model)/24981069))
print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR ROOT_MSE | NORMAL_LOSS MEAN MED <11.25 <22.5 <30\n')

# define dataset path
dataset_path = opt.dataroot

if opt.ssl_type == 'onelabel':
    labels_weights = torch.load('{}onelabel.pth'.format(opt.labelroot))['labels_weights'].float().cuda()
elif opt.ssl_type == 'randomlabels':
    labels_weights = torch.load('{}randomlabels.pth'.format(opt.labelroot))['labels_weights'].float().cuda()
if opt.task == 'semantic':
    task_index = 0
elif opt.task == 'depth':
    task_index = 1
elif opt.task == 'normal':
    task_index = 2
nyuv2_train_set = NYUv2_crop(root=dataset_path, train=True, augmentation=True)
nyuv2_test_set = NYUv2(root=dataset_path, train=False)

batch_size = 2
nyuv2_train_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_train_set,
    batch_size=batch_size,
    shuffle=True, num_workers=0, drop_last=False)

nyuv2_test_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_test_set,
    batch_size=batch_size,
    shuffle=False, num_workers=0)


# define parameters
total_epoch = 200
train_batch = len(nyuv2_train_loader)
test_batch = len(nyuv2_test_loader)
avg_cost = np.zeros([total_epoch, 24], dtype=np.float32)
lambda_weight = np.zeros([len(tasks), total_epoch])
best_performance = - 100
isbest=False
for epoch in range(total_epoch):
    index = epoch
    cost = np.zeros(24, dtype=np.float32)
    scheduler.step()
    lambda_weight[task_index, index] = 1.0

    bar = Bar('Training', max=train_batch)

    # iteration for all batches
    model.train()
    nyuv2_train_dataset = iter(nyuv2_train_loader)
    cost_seg = AverageMeter()
    cost_depth = AverageMeter()
    cost_normal = AverageMeter()
    for k in range(train_batch):
        train_data, train_label, train_depth, train_normal, image_index = nyuv2_train_dataset.next()
        train_data, train_label = train_data.cuda(), train_label.type(torch.LongTensor).cuda()
        train_depth, train_normal = train_depth.cuda(), train_normal.cuda()
        

        train_pred, logsigma, feat = model(train_data)
        loss = 0
        w_record = 0
        for ind_ in range(len(image_index)):
            if opt.ssl_type == 'full':
                w__ = torch.ones(len(tasks)).cuda()
                w = torch.ones(len(tasks)).cuda()
            else:
                w__ = labels_weights[image_index[ind_]].clone()
                w = labels_weights[image_index[ind_]]
            w_ = w__[task_index]
            w_record = w_record + w_/len(image_index)
            train_loss_ = model.model_fit(train_pred[0][ind_].unsqueeze(0), train_label[ind_].unsqueeze(0), train_pred[1][ind_].unsqueeze(0), train_depth[ind_].unsqueeze(0), train_pred[2][ind_].unsqueeze(0), train_normal[ind_].unsqueeze(0))
            if w_ == 1:
                loss = loss + w_.data * train_loss_[task_index] / len(image_index)
        train_loss = model.model_fit(train_pred[0], train_label, train_pred[1], train_depth, train_pred[2], train_normal)

        optimizer.zero_grad()
        if w_record != 0:
            loss.backward()
            optimizer.step()
        
        cost_seg.update(train_loss[0].item(), train_data.size(0))
        cost_depth.update(train_loss[1].item(), train_data.size(0))
        cost_normal.update(train_loss[2].item(), train_data.size(0))
        cost[0] = train_loss[0].item()
        cost[1] = model.compute_miou(train_pred[0], train_label).item()
        cost[2] = model.compute_iou(train_pred[0], train_label).item()
        cost[3] = train_loss[1].item()
        cost[4], cost[5] = model.depth_error(train_pred[1], train_depth)
        cost[6] = train_loss[2].item()
        cost[7], cost[8], cost[9], cost[10], cost[11] = model.normal_error(train_pred[2], train_normal)
        avg_cost[index, :12] += cost[:12] / train_batch

        if opt.task =='semantic':
            bar.suffix  = 'Epoch {epoch} : ({batch}/{size}) | LossS: {loss_s:.4f}'.format(
                    epoch=epoch+1,
                    batch=k + 1,
                    size=train_batch,
                    loss_s=cost_seg.avg,
                    )
        elif opt.task == 'depth':
            bar.suffix  = 'Epoch {epoch} : ({batch}/{size}) | LossD: {loss_d:.4f}'.format(
                    epoch=epoch+1,
                    batch=k + 1,
                    size=train_batch,
                    loss_d=cost_depth.avg,
                    )
        elif opt.task == 'normal':
            bar.suffix  = 'Epoch {epoch} : ({batch}/{size}) | LossN: {loss_n:.4f}'.format(
                    epoch=epoch+1,
                    batch=k + 1,
                    size=train_batch,
                    loss_n=cost_normal.avg,
                    )
        bar.next()
    bar.finish()

    # evaluate in the last 20 epochs
    if opt.eval_last20 == 0:
        evaluate = True
    elif opt.eval_last20 and (epoch+1) > (total_epoch - 20):
        evaluate = True
    else:
        evaluate = False

    if evaluate:
        # evaluating test data
        model.eval()
        conf_mat = ConfMatrix(model.class_nb)
        depth_mat = DepthMeter()
        normal_mat = NormalsMeter()
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
                cost[12] = test_loss[0].item()
                cost[15] = test_loss[1].item()
                cost[18] = test_loss[2].item()
                avg_cost[index, 12:] += cost[12:] / test_batch
            avg_cost[index, 13:15] = conf_mat.get_metrics()
            depth_metric = depth_mat.get_score()
            avg_cost[index, 16], avg_cost[index, 17] = depth_metric['l1'], depth_metric['rmse']
            normal_metric = normal_mat.get_score()
            avg_cost[index, 19], avg_cost[index, 20], avg_cost[index, 21], avg_cost[index, 22], avg_cost[index, 23] = normal_metric['mean'], normal_metric['rmse'], normal_metric['11.25'], normal_metric['22.5'], normal_metric['30']

        if opt.task == 'semantic':
            print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f}'
              'TEST: {:.4f} {:.4f} {:.4f}'
              .format(index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], 
                    avg_cost[index, 12], avg_cost[index, 13], avg_cost[index, 14]))
        elif opt.task == 'depth':
            print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f}'
              'TEST: {:.4f} {:.4f} {:.4f}'
              .format(index, avg_cost[index, 3], avg_cost[index, 4], avg_cost[index, 5], 
                    avg_cost[index, 15], avg_cost[index, 16], avg_cost[index, 17]))
        elif opt.task == 'normal':
            print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '
              'TEST: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'
              .format(index, avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8], avg_cost[index, 9], avg_cost[index, 10], avg_cost[index, 11],
                    avg_cost[index, 18], avg_cost[index, 19], avg_cost[index, 20], avg_cost[index, 21], avg_cost[index, 22], avg_cost[index, 23]))


        if task_index == 0:
            stl_performance = avg_cost[index, 13]
        elif task_index == 1:
            stl_performance = - avg_cost[index, 16]
        elif task_index == 2:
            stl_performance = - avg_cost[index, 19]
        isbest = stl_performance > best_performance

    logger.append([index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 3],
                avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8], avg_cost[index, 9],
                avg_cost[index, 10], avg_cost[index, 11], avg_cost[index, 12], avg_cost[index, 13],
                avg_cost[index, 14], avg_cost[index, 15], avg_cost[index, 16], avg_cost[index, 17], avg_cost[index, 18],
                avg_cost[index, 19], avg_cost[index, 20], avg_cost[index, 21], avg_cost[index, 22], avg_cost[index, 23],
                lambda_weight[0, index], lambda_weight[1, index], lambda_weight[2, index]])

    if isbest:
        best_performance = stl_performance
        print_index = index
    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_performance': best_performance,
            'optimizer' : optimizer.state_dict(),
        }, isbest) 
print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '
          'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'
          .format(print_index, avg_cost[print_index, 0], avg_cost[print_index, 1], avg_cost[print_index, 2], avg_cost[print_index, 3],
                avg_cost[print_index, 4], avg_cost[print_index, 5], avg_cost[print_index, 6], avg_cost[print_index, 7], avg_cost[print_index, 8], avg_cost[print_index, 9],
                avg_cost[print_index, 10], avg_cost[print_index, 11], avg_cost[print_index, 12], avg_cost[print_index, 13],
                avg_cost[print_index, 14], avg_cost[print_index, 15], avg_cost[print_index, 16], avg_cost[print_index, 17], avg_cost[print_index, 18],
                avg_cost[print_index, 19], avg_cost[print_index, 20], avg_cost[print_index, 21], avg_cost[print_index, 22], avg_cost[print_index, 23]))
