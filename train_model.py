# !/usr/bin/env Python
# coding = utf-8

import os
import sys
import argparse
import inspect
import datetime
import json
import numpy as np
import time
import paddle.fluid as fluid
import flow_2d_resnets
from hmdb_dataset import HMDB as DS
from paddle.fluid.layers import mean, reshape

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow', default='rgb')
parser.add_argument('-model', type=str, help='', default='2d')
parser.add_argument('-exp_name', type=str, default='hmdb1')
parser.add_argument('-batch_size', type=int, default=24)#64
parser.add_argument('-length', type=int, default=16)
parser.add_argument('-learnable', type=str, default='[0,1,1,1,1]')
parser.add_argument('-niter', type=int, default=20)
parser.add_argument('-use_gpu', type=bool, default=True)
parser.add_argument('-pretrain', type=str, default=None, help='path to pretrain weights')
parser.add_argument('-save_dir', type=str, default=None, help='path to save train snapshoot')

args = parser.parse_args()


#import models

place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
# 进入paddle动态图环境
with fluid.dygraph.guard(place):
    ##################
    #system
    # Create model, dataset, and training setup
    #
    ##################
    # 定义模型
    model = flow_2d_resnets.resnet50(pretrained=False, mode=args.mode, n_iter=args.niter, learnable=eval(args.learnable), num_classes=51)
    if args.pretrain is not None:
        model_arg, _ = fluid.dygraph.load_dygraph(args.pretrain)
        model.load_dict(model_arg)
    # 优化器
    clip = fluid.clip.GradientClipByGlobalNorm(clip_norm=1.0)
    opt = fluid.optimizer.Momentum(learning_rate=0.003, momentum=0.9, parameter_list=model.parameters(),regularization=fluid.regularizer.L2Decay(1e-3))  # , grad_clip=clip
    #opt = fluid.optimizer.SGD(learning_rate = 0.01, parameter_list=model.parameters(),regularization=fluid.regularizer.L2Decay(1e-3))
    # opt = fluid.optimizer.AdamOptimizer(0.003, 0.9,parameter_list=model.parameters(), regularization=fluid.regularizer.L2Decay(regularization_coeff=1e-6),epsilon=1e-8)
    # 批大小batch_size，根据显卡设定
    batch_size = args.batch_size
    
    dataseta = DS(hmdb_pth='hmdb_dtst/', pkl_pth='train/', model=args.model, mode=args.mode, length=args.length, batch_size=batch_size)
    dl = dataseta.create_reader()

    dataset = DS(hmdb_pth='hmdb_dtst/', pkl_pth='test/', model=args.model, mode=args.mode, length=args.length, batch_size=batch_size) 
    vdl = dataset.create_reader()
    dataloader = {'train': dl, 'val': vdl}


    #################
    #
    # Setup logs, store model code
    # hyper-parameters, etc...
    #
    #################

    log_name = datetime.datetime.today().strftime('%m-%d-%H%M')+'-'+args.exp_name
    log_path = os.path.join('logs/',log_name)
    os.mkdir(log_path)
    os.system('cp * logs/'+log_name+'/')

    # deal with hyper-params...
    with open(os.path.join(log_path,'params.json'), 'w') as out:
        hyper = vars(args)
        json.dump(hyper, out)
    log = {'iterations': [], 'epoch':[], 'validation':[], 'train_acc':[], 'val_acc':[]}

    ###############
    #
    # Train the model and save everything
    # 正式训练
    #
    ###############
    num_epochs = 60
    top_acc = 0.
    for epoch in range(num_epochs):
        for phase in ['train']:  # , 'val'
            train = (phase == 'train')
            if phase == 'train':
                model.train()
            else:
                model.eval()
            tloss = 0.
            acc = 0.
            tot = 0
            e = s = 0

            for batch_id,data in enumerate(dataloader[phase]()):
                vid = np.array([x[0] for x in data]).astype('float32')  # 视频帧 BTCHW
                cls = np.array([x[1] for x in data]).astype('int64')  # 类别编号

                s = time.time()
                # print('vid_shape: ', np.shape(vid))

                # 源数据转tensor
                vid = fluid.dygraph.to_variable(vid)
                cls = fluid.dygraph.to_variable(cls)
                cls = reshape(cls,shape=[-1,1])
                # forward
                out, acc = model(vid, cls)
                '''
                pred = torch.max(outputs, dim=1)[1]
                corr = torch.sum((pred == cls).int())
                acc += corr.item()
                tot += vid.size(0)
                '''
                # loss
                loss = fluid.layers.cross_entropy(out, cls)
                avg_loss = mean(loss)
                # print('loss: ', float(avg_loss))
                if phase == 'train':
                    # backward
                    avg_loss.backward()
                    opt.minimize(avg_loss)
                    model.clear_gradients()
                e = time.time()
                print('epoch', epoch, ' phase', phase, ' batch', batch_id, ' time', (e-s), ' loss', float(avg_loss), ' acc', acc)
                if acc>=top_acc:
                    top_acc = acc
                    if args.save_dir is not None:
                        fluid.dygraph.save_dygraph(model.state_dict(),args.save_dir+'/epoch{}batch{}'.format(epoch, batch_id))

            # 学习率调整
            # if phase == 'eval':
            #    lr_sched.step(tloss/c)

        with open(os.path.join(log_path,'log.json'), 'w') as out:
            json.dump(log, out)
        


        #lr_sched.step()
