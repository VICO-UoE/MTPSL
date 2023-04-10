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
from dataset.cityscapesssl import *
from torch.autograd import Variable
from model.segnet_mtl_cityscapes import SegNet
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
parser.add_argument('--dataroot', default='./data/cityscapes', type=str, help='dataset root')
parser.add_argument('--out', default='./results/cityscapes', help='Directory to output the result')
parser.add_argument('--ssl-type', default='randomlabels', type=str, help='ssl type: onelabel, randomlabels, full')
parser.add_argument('--labelroot', default='./data/cityscapes_settings/', type=str, help='partially setting root')
parser.add_argument('--eval-last20', default=0, type=int, help='1 means we evaluate models in the last 20 epochs')

opt = parser.parse_args()

tasks = ['semantic', 'depth']

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

title = 'Cityscapes'
logger = Logger(os.path.join(opt.out, 'stl_sl_{}_{}_log.txt'.format(opt.ssl_type, opt.task)), title=title)
logger.set_names(['Epoch', 'T.Ls', 'T. mIoU', 'T. Pix', 'T.Ld', 'T.abs', 'T.rel',
    'V.Ls', 'V. mIoU', 'V. Pix', 'V.Ld', 'V.abs', 'V.rel', 'Ws', 'Wd'])

# define model, optimiser and scheduler
model = SegNet(type_=opt.type, class_nb=7).cuda()
params = []
params += model.parameters()
optimizer = optim.Adam(params, lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)


# compute parameter space
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print('Parameter Space: ABS: {:.1f}, REL: {:.4f}\n'.format(count_parameters(model),
                                                           count_parameters(model)/24981069))
print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR ROOT_MSE\n')

# define dataset path
dataset_path = opt.dataroot

if opt.ssl_type == 'onelabel':
    labels_weights = torch.load('{}onelabel.pth'.format(opt.labelroot))['labels_weights'].float().cuda()
if opt.task == 'semantic':
    task_index = 0
elif opt.task == 'depth':
    task_index = 1
cityscapes_train_set = Cityscapes_crop(root=dataset_path, train=True, augmentation=True, flip=True)
cityscapes_test_set = Cityscapes(root=dataset_path, train=False)

batch_size = 8
cityscapes_train_loader = torch.utils.data.DataLoader(
    dataset=cityscapes_train_set,
    batch_size=batch_size,
    shuffle=True, num_workers=4, drop_last=False)

cityscapes_test_loader = torch.utils.data.DataLoader(
    dataset=cityscapes_test_set,
    batch_size=batch_size,
    shuffle=False, num_workers=4)


# define parameters
total_epoch = 200
train_batch = len(cityscapes_train_loader)
test_batch = len(cityscapes_test_loader)
avg_cost = np.zeros([total_epoch, 12], dtype=np.float32)
lambda_weight = np.zeros([len(tasks), total_epoch])
best_performance = - 100
isbest=False
for epoch in range(total_epoch):
    index = epoch
    cost = np.zeros(12, dtype=np.float32)
    scheduler.step()
    lambda_weight[task_index, index] = 1.0

    bar = Bar('Training', max=train_batch)

    # iteration for all batches
    model.train()
    cityscapes_train_dataset = iter(cityscapes_train_loader)
    cost_seg = AverageMeter()
    cost_depth = AverageMeter()
    for k in range(train_batch):
        train_data, train_label, train_depth, image_index = cityscapes_train_dataset.next()
        train_data, train_label = train_data.cuda(), train_label.type(torch.LongTensor).cuda()
        train_depth = train_depth.cuda()
        

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
            train_loss_ = model.model_fit(train_pred[0][ind_].unsqueeze(0), train_label[ind_].unsqueeze(0), train_pred[1][ind_].unsqueeze(0), train_depth[ind_].unsqueeze(0))
            if w_ == 1:
                loss = loss + w_.data * train_loss_[task_index] / len(image_index)
        train_loss = model.model_fit(train_pred[0], train_label, train_pred[1], train_depth)
        optimizer.zero_grad()
        if w_record != 0:
            loss.backward()
            optimizer.step()
        
        cost_seg.update(train_loss[0].item(), train_data.size(0))
        cost_depth.update(train_loss[1].item(), train_data.size(0))
        cost[0] = train_loss[0].item()
        cost[1] = model.compute_miou(train_pred[0], train_label).item()
        cost[2] = model.compute_iou(train_pred[0], train_label).item()
        cost[3] = train_loss[1].item()
        cost[4], cost[5] = model.depth_error(train_pred[1], train_depth)
        avg_cost[index, :6] += cost[:6] / train_batch

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
        with torch.no_grad():  # operations inside don't track history
            cityscapes_test_dataset = iter(cityscapes_test_loader)
            for k in range(test_batch):
                test_data, test_label, test_depth = cityscapes_test_dataset.next()
                test_data, test_label = test_data.cuda(),  test_label.type(torch.LongTensor).cuda()
                test_depth = test_depth.cuda()

                test_pred, _, _ = model(test_data)
                test_loss = model.model_fit(test_pred[0], test_label, test_pred[1], test_depth)

                conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())
                depth_mat.update(test_pred[1], test_depth)
                cost[6] = test_loss[0].item()
                cost[9] = test_loss[1].item()
                avg_cost[index, 6:] += cost[6:] / test_batch
            avg_cost[index, 7:9] = conf_mat.get_metrics()
            depth_metric = depth_mat.get_score()
            avg_cost[index, 10], avg_cost[index, 11] = depth_metric['l1'], depth_metric['rmse']

        if opt.task == 'semantic':
            print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} '
              'TEST: {:.4f} {:.4f} {:.4f}'
              .format(index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], 
                    avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8]))
        elif opt.task == 'depth':
            print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} '
              'TEST: {:.4f} {:.4f} {:.4f}'
              .format(index, avg_cost[index, 3], avg_cost[index, 4], avg_cost[index, 5], 
                    avg_cost[index, 9], avg_cost[index, 10], avg_cost[index, 11]))


        if task_index == 0:
            stl_performance = avg_cost[index, 7]
        elif task_index == 1:
            stl_performance = - avg_cost[index, 10]
        isbest = stl_performance > best_performance

    logger.append([index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 3],
                avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8], avg_cost[index, 9],
                avg_cost[index, 10], avg_cost[index, 11],
                lambda_weight[0, index], lambda_weight[1, index]])

    if isbest:
        best_performance = stl_performance
        print_index = index
    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_performance': best_performance,
            'optimizer' : optimizer.state_dict(),
        }, isbest) 
print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} '
          'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f}'
          .format(print_index, avg_cost[print_index, 0], avg_cost[print_index, 1], avg_cost[print_index, 2], avg_cost[print_index, 3],
                avg_cost[print_index, 4], avg_cost[print_index, 5], avg_cost[print_index, 6], avg_cost[print_index, 7], avg_cost[print_index, 8], avg_cost[print_index, 9],
                avg_cost[print_index, 10], avg_cost[print_index, 11]))
