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

parser = argparse.ArgumentParser(description='Multi-task partially-supervised learning with cross-task consistency (SegNet)')
parser.add_argument('--type', default='standard', type=str, help='split type: standard, wide, deep')
parser.add_argument('--weight', default='uniform', type=str, help='multi-task weighting: uniform')
parser.add_argument('--dataroot', default='./data/nyuv2', type=str, help='dataset root')
parser.add_argument('--temp', default=1.0, type=float, help='temperature for DWA (must be positive)')
parser.add_argument('--wlr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--out', default='./results/nyuv2', help='Directory to output the result')
parser.add_argument('--alpha', default=1.5, type=float, help='hyper params of GradNorm')
parser.add_argument('--ssl-type', default='randomlabels', type=str, help='ssl type: onelabel, randomlabels, full')
parser.add_argument('--labelroot', default='./data/nyuv2_settings/', type=str, help='partially setting root')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--eval-last20', default=0, type=int, help='1 means we evaluate models in the last 20 epochs')
parser.add_argument('--rampup', default='fixed', type=str, help='up for ramp-up loss weight of cross-task consistency loss, fixed use constant loss weight.')
parser.add_argument('--con-weight', default=2.0, type=float, help='weight for cross-task consistency loss')
parser.add_argument('--reg-weight', default=0.5, type=float, help='weight for cross-task consistency loss')

opt = parser.parse_args()

tasks = ['semantic', 'depth', 'normal']
input_channels = [13, 1, 3]

if not os.path.isdir(opt.out):
    mkdir_p(opt.out)
paths = [opt.ssl_type, 'mtl']
for i in range(len(paths)):
    opt.out = os.path.join(opt.out, paths[i])
    if not os.path.isdir(opt.out):
        mkdir_p(opt.out)

stl_performance = {
                    'full': {'semantic': 37.447399999999995, 'depth': 0.607902, 'normal': 25.938105}, 
                    'onelabel': {'semantic': 26.1113, 'depth': 0.771502, 'normal': 30.073763}, 
                    'randomlabels': {'semantic': 28.7153, 'depth': 0.754012, 'normal': 28.946388}
}


def save_checkpoint(state, is_best, checkpoint=opt.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, 'mtl_xtc_{}_{}_{}_{}_'.format(opt.ssl_type, opt.rampup, opt.con_weight, opt.reg_weight) + filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'mtl_xtc_{}_{}_{}_{}_'.format(opt.ssl_type, opt.rampup, opt.con_weight, opt.reg_weight) + 'model_best.pth.tar'))


title = 'NYUv2'
logger = Logger(os.path.join(opt.out, 'mtl_xtc_{}_{}_{}_{}_log.txt'.format(opt.ssl_type, opt.rampup, opt.con_weight, opt.reg_weight)), title=title)
logger.set_names(['Epoch', 'T.Ls', 'T. mIoU', 'T. Pix', 'T.Ld', 'T.abs', 'T.rel', 'T.Ln', 'T.Mean', 'T.Med', 'T.11', 'T.22', 'T.30',
    'V.Ls', 'V. mIoU', 'V. Pix', 'V.Ld', 'V.abs', 'V.rel', 'V.Ln', 'V.Mean', 'V.Med', 'V.11', 'V.22', 'V.30', 'Con L', 'Ws', 'Wd', 'Wn'])

# define model, optimiser and scheduler
model = SegNet(type_=opt.type, class_nb=13).cuda()
mapfns = Mapfns(tasks=tasks, input_channels=input_channels).cuda()
params = []
params += model.parameters()

params += [v for k, v in mapfns.named_parameters() if 'gamma' not in k and 'beta' not in k]
optimizer = optim.Adam(params, lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
params_film = [v for k, v in mapfns.named_parameters() if 'gamma' in k or 'beta' in k]
# optimizer for the conditional auxiliary network
optimizer_film = optim.Adam(params_film, lr=1e-3)
scheduler_film = optim.lr_scheduler.StepLR(optimizer_film, step_size=30, gamma=0.5)
start_epoch = 0
if opt.resume:
    checkpoint = torch.load(opt.resume)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    start_epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optimizer'])
    optimizer_film.load_state_dict(checkpoint['optimizer_film'])
    print('=> checkpoint from {} loaded!'.format(opt.resume))


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
nyuv2_train_set = NYUv2_crop(root=dataset_path, train=True, augmentation=True, aug_twice=True)
nyuv2_test_set = NYUv2(root=dataset_path, train=False)

batch_size = 2
nyuv2_train_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_train_set,
    batch_size=batch_size,
    shuffle=True, num_workers=0, drop_last=True)

nyuv2_test_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_test_set,
    batch_size=batch_size,
    shuffle=False, num_workers=0)


# define parameters
total_epoch = 200
train_batch = len(nyuv2_train_loader)
test_batch = len(nyuv2_test_loader)
T = opt.temp
avg_cost = np.zeros([total_epoch, 24], dtype=np.float32)
ctl_cost = np.zeros([total_epoch, 1], dtype=np.float32)
lambda_weight = np.zeros([3, total_epoch])
best_performance = -100
isbest=False

for epoch in range(start_epoch, total_epoch):
    index = epoch
    print('lr at {}th epoch is {} for optimizer and {} for film'.format(index, optimizer.param_groups[0]['lr'], optimizer_film.param_groups[0]['lr']))
    cost = np.zeros(24, dtype=np.float32)

    # apply Dynamic Weight Average
    if opt.weight == 'dwa':
        if index == 0 or index == 1:
            lambda_weight[:, index] = 1.0
        else:
            w_1 = avg_cost[index - 1, 0] / avg_cost[index - 2, 0]
            w_2 = avg_cost[index - 1, 3] / avg_cost[index - 2, 3]
            w_3 = avg_cost[index - 1, 6] / avg_cost[index - 2, 6]
            lambda_weight[0, index] = 3 * np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))
            lambda_weight[1, index] = 3 * np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))
            lambda_weight[2, index] = 3 * np.exp(w_3 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))

    bar = Bar('Training', max=train_batch)

    # iteration for all batches
    model.train()
    mapfns.train()

    con_loss_ave = AverageMeter()
    cost_seg = AverageMeter()
    cost_depth = AverageMeter()
    cost_normal = AverageMeter()
    nyuv2_train_dataset = iter(nyuv2_train_loader)
    for k in range(train_batch):
        train_data, train_label, train_depth, train_normal, image_index, train_data1, train_label1, train_depth1, train_normal1, trans_params = nyuv2_train_dataset.next()
        train_data, train_label = train_data.cuda(), train_label.type(torch.LongTensor).cuda()
        train_depth, train_normal = train_depth.cuda(), train_normal.cuda()
        train_data1, train_label1 = train_data1.cuda(), train_label1.type(torch.LongTensor).cuda()
        train_depth1, train_normal1 = train_depth1.cuda(), train_normal1.cuda()
        

        train_data_ = torch.cat([train_data, train_data1], dim=0)
        train_pred, logsigma, feat = model(train_data_)
        feat_aug = feat[0][batch_size:]
        feat = feat[0][:batch_size]
        train_pred_aug = [train_pred[0][batch_size:], train_pred[1][batch_size:], train_pred[2][batch_size:]]
        train_pred = [train_pred[0][:batch_size], train_pred[1][:batch_size], train_pred[2][:batch_size]]
        loss = 0
        for ind_ in range(len(image_index)):
            if opt.ssl_type == 'full':
                w = torch.ones(len(tasks)).float().cuda()
            else:
                w = labels_weights[image_index[ind_]].clone().float().cuda()
            train_pred_seg = train_pred_aug[0][ind_][None,:,:,:]
            train_pred_depth = train_pred_aug[1][ind_][None,:,:,:]
            train_pred_normal = train_pred_aug[2][ind_][None,:,:,:]
            _sc, _h, _w, _i, _j, height, width = trans_params[ind_]
            _h, _w, _i, _j, height, width = int(_h), int(_w), int(_i), int(_j), int(height), int(width)
            
            train_target_ind = [train_label1[ind_].unsqueeze(0), train_depth1[ind_].unsqueeze(0), train_normal1[ind_].unsqueeze(0)]
            train_loss_ind = model.model_fit(train_pred[0][ind_].unsqueeze(0), train_label[ind_].unsqueeze(0), train_pred[1][ind_].unsqueeze(0), train_depth[ind_].unsqueeze(0), train_pred[2][ind_].unsqueeze(0), train_normal[ind_].unsqueeze(0))
            for i in range(len(tasks)):
                if w[i] == 0:
                    train_loss_ind[i] = 0
            train_pred_ind = [train_pred_seg, train_pred_depth, train_pred_normal]

            # compute the cross-task consistency loss
            con_loss = mapfns(train_pred_ind, train_target_ind, feat_aug[ind_].unsqueeze(0), copy.deepcopy(w), ssl_type=opt.ssl_type)

            if opt.rampup == 'up':
                if epoch > 99:
                    con_weight = 1
                else:
                    con_weight = (k/train_batch + epoch) / 100
            else:
                con_weight = 1
            con_weight *= opt.con_weight

            con_loss_ave.update(con_loss.item(), 1)
            loss = loss + sum(train_loss_ind[i] for i in range(len(tasks))) / len(image_index) + con_loss * con_weight / len(image_index)
        train_loss = model.model_fit(train_pred[0], train_label, train_pred[1], train_depth, train_pred[2], train_normal)

        optimizer.zero_grad()
        optimizer_film.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_film.step()
        
        cost_seg.update(train_loss[0].item(), batch_size)
        cost_depth.update(train_loss[1].item(), batch_size)
        cost_normal.update(train_loss[2].item(), batch_size)
        cost[0] = train_loss[0].item()
        cost[1] = model.compute_miou(train_pred[0], train_label).item()
        cost[2] = model.compute_iou(train_pred[0], train_label).item()
        cost[3] = train_loss[1].item()
        cost[4], cost[5] = model.depth_error(train_pred[1], train_depth)
        cost[6] = train_loss[2].item()
        cost[7], cost[8], cost[9], cost[10], cost[11] = model.normal_error(train_pred[2], train_normal)
        avg_cost[index, :12] += cost[:12] / train_batch
        ctl_cost[index, 0] += con_loss / train_batch
        bar.suffix  = '({batch}/{size}) | LossS: {loss_s:.4f} | LossD: {loss_d:.4f} | LossN: {loss_n:.4f} | Ws: {ws:.4f} | Wd: {wd:.4f}| Wn: {wn:.4f} | CTL: {ctl:.4f} | CW: {cw:.2f}'.format(
                    batch=k + 1,
                    size=train_batch,
                    # loss_s=cost[1],
                    # loss_d=cost[3],
                    # loss_n=cost[6],
                    loss_s=cost_seg.avg,
                    loss_d=cost_depth.avg,
                    loss_n=cost_normal.avg,
                    ws=w[0].data,
                    wd=w[1].data,
                    wn=w[2].data,
                    ctl=con_loss_ave.avg,
                    cw=con_weight,
                    )
        bar.next()
    bar.finish()

    

    if opt.eval_last20 == 0:
        evaluate = True
    elif opt.eval_last20 and (epoch + 1) > (total_epoch - 20):
        evaluate = True
    else:
        evaluate = False

    # evaluating test data
    if evaluate:
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
        scheduler.step()
        scheduler_film.step()

        mtl_performance = 0.0
        mtl_performance += (avg_cost[index, 13]* 100 - stl_performance[opt.ssl_type]['semantic']) / stl_performance[opt.ssl_type]['semantic']
        mtl_performance -= (avg_cost[index, 16] - stl_performance[opt.ssl_type]['depth']) / stl_performance[opt.ssl_type]['depth']
        mtl_performance -= (avg_cost[index, 19] - stl_performance[opt.ssl_type]['normal']) / stl_performance[opt.ssl_type]['normal']
        mtl_performance = mtl_performance / len(tasks) * 100
        isbest = mtl_performance > best_performance
        print('current performance: {:.4f}, best performance: {:.4f}'.format(mtl_performance, best_performance))

        print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '
              'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'
              .format(index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 3],
                    avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8], avg_cost[index, 9],
                    avg_cost[index, 10], avg_cost[index, 11], avg_cost[index, 12], avg_cost[index, 13],
                    avg_cost[index, 14], avg_cost[index, 15], avg_cost[index, 16], avg_cost[index, 17], avg_cost[index, 18],
                    avg_cost[index, 19], avg_cost[index, 20], avg_cost[index, 21], avg_cost[index, 22], avg_cost[index, 23]))
        logger.append([index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 3],
                    avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8], avg_cost[index, 9],
                    avg_cost[index, 10], avg_cost[index, 11], avg_cost[index, 12], avg_cost[index, 13],
                    avg_cost[index, 14], avg_cost[index, 15], avg_cost[index, 16], avg_cost[index, 17], avg_cost[index, 18],
                    avg_cost[index, 19], avg_cost[index, 20], avg_cost[index, 21], avg_cost[index, 22], avg_cost[index, 23], ctl_cost[index, 0],
                    lambda_weight[0, index], lambda_weight[1, index], lambda_weight[2, index]])

    if isbest:
        best_performance = mtl_performance
        print_index = index
    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'mapfns': mapfns.state_dict(),
            'best_performance': best_performance,
            'optimizer' : optimizer.state_dict(),
            'optimizer_film': optimizer_film.state_dict(),
            'avg_cost': avg_cost,
        }, isbest) 
print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '
          'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'
          .format(print_index, avg_cost[print_index, 0], avg_cost[print_index, 1], avg_cost[print_index, 2], avg_cost[print_index, 3],
                avg_cost[print_index, 4], avg_cost[print_index, 5], avg_cost[print_index, 6], avg_cost[print_index, 7], avg_cost[print_index, 8], avg_cost[print_index, 9],
                avg_cost[print_index, 10], avg_cost[print_index, 11], avg_cost[print_index, 12], avg_cost[print_index, 13],
                avg_cost[print_index, 14], avg_cost[print_index, 15], avg_cost[print_index, 16], avg_cost[print_index, 17], avg_cost[print_index, 18],
                avg_cost[print_index, 19], avg_cost[print_index, 20], avg_cost[print_index, 21], avg_cost[print_index, 22], avg_cost[print_index, 23]))
