from __future__ import print_function, division
import sys 
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from raftKD import RAFT
import evaluate
import datasets

from torch.utils.tensorboard import SummaryWriter
from distillation import *
from adaptation import *
from datetime import datetime
import csv

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000


#def sequence_loss(flow_preds, flow_gt, valid, flow_predictions_teacher,corr_mat, corr_mat_teacher, fkd_loss=0, gamma=0.8, alpha=0.1, max_flow=MAX_FLOW):
def sequence_loss(flow_preds, flow_gt, valid, flow_predictions_teacher,corr_mat, corr_mat_teacher, fkd_loss=0, gamma=0.8, alpha=0.1, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0
    rskd_loss = 0.0
    mkd_loss = 0.0
    a, b, c = alpha, alpha, alpha
    

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)



    for i in range(n_predictions):

        #---------------------------------------------------#
        #   Adapt
        #---------------------------------------------------# 
        
        t = adapter() 
        distiller = DistillerMKD()
        # _sgn, _inertia = t.signspatternmatrix(flow_predictions_teacher[i],flow_preds[i].size(1))
        # sgn, inertia = t.signspatternmatrix(flow_preds[i],flow_preds[i].size(1))

        # _sgn_corr, _inertia_corr = t.signspatternmatrix(corr_mat_teacher[i],corr_mat[i].size(1))
        # sgn_corr, inertia_corr = t.signspatternmatrix(corr_mat[i],corr_mat[i].size(1))
        convteacher = t.adapt(flow_predictions_teacher[i],flow_preds[i].size(1))
        convcorr = t.adapt(corr_mat_teacher[i],corr_mat[i].size(1))
        corri_loss = distiller.ssim(corr_mat[i], convcorr) #+ distiller.ssim(inertia_corr, _inertia_corr) 
        convcorr.to
        di_loss = distiller.ssim(flow_preds[i], convteacher) #+ distiller.ssim(inertia, _inertia) 

        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()
        rskd_loss += i_weight * (valid[:, None] * ((1-alpha)*i_loss + alpha*di_loss)).mean()
        mkd_loss += i_weight * (valid[:, None] * (i_loss + c*fkd_loss + b*corri_loss + a*di_loss)).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics, rskd_loss, mkd_loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()

def getteachermodel(args):
    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.eval()
    return model

def train(args, teachermodel):

    model = nn.DataParallel(RAFT(args, nature="small"), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.cuda()
    model.train()

    if args.stage != 'chairs':
        model.module.freeze_bn()

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    VAL_FREQ = 5000
    add_noise = True
    epoch = 0

    should_keep_training = True
    while should_keep_training:
        #temperature = args.temperature
        distilltype = args.distilltype
        counter = 0
        

        adaptation = adapter()
        distiller = DistillerMKD()

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()

            image1, image2, flow, valid = [x.cuda() for x in data_blob]

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            flow_predictions,  corr_mat, fm1, fm2 = model(image1, image2, iters=args.iters) 
            flow_predictions_teacher,  corr_mat_teacher, fm1_teacher, fm2_teacher = teachermodel(image1, image2, iters=args.iters)  

            t = adapter()
            _sgn_fm1, _inertia1 = t.signspatternmatrix(fm1_teacher,fm1.size(1))
            sgn_fm1, inertia1 = t.signspatternmatrix(fm1,fm1.size(1))
            

            convfm1 = t.adapt(fm1_teacher, fm1.size(1))
            #fm1_loss = distiller.RE_loss(fm1, convfm1) + distiller.KL_loss(inertia1, _inertia1)
            fm1_loss = distiller.ssim(fm1, convfm1)

            _sgn_fm2, _inertia2 = adaptation.signspatternmatrix(fm2_teacher,fm2.size(1))
            sgn_fm2, inertia2 = adaptation.signspatternmatrix(fm2,fm2.size(1))
            #fm2_loss = distiller.RE_loss(sgn_fm2, _sgn_fm2) + distiller.KL_loss(inertia2, _inertia2)
            convfm2 = t.adapt(fm2_teacher, fm2.size(1))
            fm2_loss = distiller.ssim(fm2, convfm2)
            fkd_loss = (fm2_loss+fm1_loss).mean()

            studentloss, metrics, rskd_loss, mkd_loss = sequence_loss(flow_predictions, flow, valid, flow_predictions_teacher, corr_mat, corr_mat_teacher, fkd_loss, args.gamma, args.alpha)

            if distilltype=="rskd":
                print(f'RsKD === step: {counter + 1}== normal Loss {studentloss}====== distiation Loss {rskd_loss} === metrics:{metrics}')
                loss = rskd_loss

                with open("loss_rskd.csv", "a") as file:
                    csvreader = csv.reader(file)
                    writer = csv.writer(file)

                    if epoch==0 and counter ==0:
                        writer.writerow(["date", "epoch", "normal Loss", "final Loss"])
                    
                    writer.writerow([datetime.now(), epoch, studentloss.float(), loss.float])
                counter +=1

            elif distilltype=="mkd":
                print(f'MKD === step: {counter + 1}== normal Loss {studentloss}====== distiation Loss {mkd_loss} === metrics:{metrics}')
                loss = mkd_loss

                with open("loss_mkd.csv", "a") as file:
                    csvreader = csv.reader(file)
                    writer = csv.writer(file)

                    if epoch==0 and counter ==0:
                        writer.writerow(["date", "epoch", "normal Loss", "final Loss"])
                    
                    writer.writerow([datetime.now(), epoch, studentloss.float(), loss])
                counter +=1

            else:
                print(f'No distill === step: {counter + 1} === metrics:{metrics}== normal Loss {studentloss}')
                loss = studentloss
                counter +=1

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
                torch.save(model.state_dict(), PATH)

                results = {}
                for val_dataset in args.validation:
                    if val_dataset == 'chairs':
                        results.update(evaluate.validate_chairs(model.module))
                    elif val_dataset == 'sintel':
                        results.update(evaluate.validate_sintel(model.module))
                    elif val_dataset == 'kitti':
                        results.update(evaluate.validate_kitti(model.module))

                logger.write_dict(results)
                
                model.train()
                if args.stage != 'chairs':
                    model.module.freeze_bn()
            
            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break
        epoch +=1
    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)
    

    return PATH


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    argss = args
    # Setup teacher
    teachermodel = getteachermodel(args)

    # Setup student
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', default='sintel', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--nature', default='small', help='use small model')
    parser.add_argument('--distilltype', default='mkd', help='choose distillation type')
    parser.add_argument('--validation', default='sintel', type=str, nargs='+')
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--num_steps', type=int, default=50000)
    parser.add_argument('--batch_size', type=int,default=1) # Normal =6
    parser.add_argument('--image_size', type=int, nargs='+', default=[368, 496])
    parser.add_argument('--iters', type=int, default=10) 
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')

    argss = parser.parse_args()
    print(argss)


    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(argss, teachermodel)