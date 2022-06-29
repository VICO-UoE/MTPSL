import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np
import pdb

# Updated evaluation metrics from https://github.com/lorenmt/mtan and from the SOTA paper https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch

class ConfMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, pred, target):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def get_metrics(self):
        h = self.mat.float()
        acc = torch.diag(h).sum() / h.sum()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return torch.mean(iu).cpu().numpy(), acc.cpu().numpy()

class NormalsMeter(object):
    def __init__(self):
        self.eval_dict = {'mean': 0., 'rmse': 0., '11.25': 0., '22.5': 0., '30': 0., 'n': 0}

    @torch.no_grad()
    def update(self, pred, gt):
        # Performance measurement happens in pixel wise fashion (Same as code from ASTMT (above))
        valid_mask = (torch.sum(gt, dim=1) != 0)
        invalid_mask = (torch.sum(gt, dim=1) == 0)
        
        # Calculate difference expressed in degrees 
        deg_diff_tmp = (180 / math.pi) * (torch.acos(torch.clamp(torch.sum(pred * gt, 1).masked_select(valid_mask), min=-1, max=1)))

        self.eval_dict['mean'] += torch.sum(deg_diff_tmp).item()
        self.eval_dict['rmse'] += torch.sum(torch.pow(deg_diff_tmp, 2)).item()
        self.eval_dict['11.25'] += torch.sum((deg_diff_tmp < 11.25).float()).item() * 100
        self.eval_dict['22.5'] += torch.sum((deg_diff_tmp < 22.5).float()).item() * 100
        self.eval_dict['30'] += torch.sum((deg_diff_tmp < 30).float()).item() * 100
        self.eval_dict['n'] += deg_diff_tmp.numel()

    def reset(self):
        self.eval_dict = {'mean': 0., 'rmse': 0., '11.25': 0., '22.5': 0., '30': 0., 'n': 0}

    def get_score(self, verbose=True):
        eval_result = dict()
        eval_result['mean'] = self.eval_dict['mean'] / self.eval_dict['n']
        eval_result['rmse'] = np.sqrt(self.eval_dict['rmse'] / self.eval_dict['n'])
        eval_result['11.25'] = self.eval_dict['11.25'] / self.eval_dict['n']
        eval_result['22.5'] = self.eval_dict['22.5'] / self.eval_dict['n']
        eval_result['30'] = self.eval_dict['30'] / self.eval_dict['n']

        return eval_result


class DepthMeter(object):
    def __init__(self):
        self.total_rmses = 0.0
        self.total_l1 = 0.0
        self.total_log_rmses = 0.0
        self.n_valid = 0.0
        self.num_images = 0.0
        self.n_valid_image = []
        self.rmses = []

    @torch.no_grad()
    def update(self, pred, gt):
        pred, gt = pred.squeeze(), gt.squeeze()
        self.num_images += pred.size(0)
        
        # Determine valid mask
        mask = (gt != 0).bool()
        self.n_valid += mask.float().sum().item() # Valid pixels per image
        
        # Only positive depth values are possible
        pred = torch.clamp(pred, min=1e-9)

        # Per pixel rmse and log-rmse.
        log_rmse_tmp = torch.pow(torch.log(gt) - torch.log(pred), 2)
        log_rmse_tmp = torch.masked_select(log_rmse_tmp, mask)
        self.total_log_rmses += log_rmse_tmp.sum().item()

        pred = pred.masked_select(mask)
        gt = gt.masked_select(mask)
        rmse_tmp = (gt-pred).abs().pow(2).cpu()

        l1_tmp = (gt-pred).abs()
        self.total_rmses += rmse_tmp.sum().item()
        self.total_l1 += l1_tmp.sum().item()

    def reset(self):
        self.rmses = []
        self.log_rmses = []
        
    def get_score(self, verbose=True):
        eval_result = dict()
        eval_result['rmse'] = np.sqrt(self.total_rmses / self.n_valid)
        eval_result['l1'] = self.total_l1 / self.n_valid
        eval_result['log_rmse'] = np.sqrt(self.total_log_rmses / self.n_valid)

        return eval_result