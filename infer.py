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
parser.add_argument('-batch_size', type=int, default=25)  # 64
parser.add_argument('-length', type=int, default=16)

parser.add_argument('-niter', type=int, default=20)
parser.add_argument('-use_gpu', type=bool, default=True)
parser.add_argument('-pretrain', type=str, default=None, help='path to pretrain weights')

args = parser.parse_args()

assert args.pretrain is not None, '请给权重文件'
# import models

place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
# 进入paddle动态图环境
with fluid.dygraph.guard(place):
    ##################
    # system
    # Create model, dataset, and training setup
    #
    ##################
    # 定义模型
    model = flow_2d_resnets.resnet50(pretrained=False, mode=args.mode, n_iter=args.niter, num_classes=51)
    model_arg, _ = fluid.dygraph.load_dygraph(args.pretrain)
    model.load_dict(model_arg)

    # 批大小batch_size，根据显卡设定
    batch_size = args.batch_size

    dataset = DS(hmdb_pth='hmdb_dtst/', pkl_pth='test/', model=args.model, mode=args.mode, length=args.length,
                 batch_size=batch_size)  # c2i=dataseta.class_to_id
    # vdl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    vdl = dataset.create_reader()

    #################
    #
    # Setup logs, store model code
    # hyper-parameters, etc...
    #
    #################

    log_name = datetime.datetime.today().strftime('%m-%d-%H%M') + '-' + args.exp_name
    log_path = os.path.join('logs/', log_name)
    os.mkdir(log_path)
    os.system('cp * logs/' + log_name + '/')

    # deal with hyper-params...
    with open(os.path.join(log_path, 'params.json'), 'w') as out:
        hyper = vars(args)
        json.dump(hyper, out)
    log = {'iterations': [], 'validation': [], 'train_acc': [], 'val_acc': []}

    ###############
    #
    # infer
    #
    ###############

    model.eval()
    tloss = 0.
    acc = 0.
    tot = 0
    e = s = 0
    accs = []
    for batch_id, data in enumerate(vdl()):
        vid = np.array([x[0] for x in data]).astype('float32')  # 视频帧 BTCHW
        cls = np.array([x[1] for x in data]).astype('int64')  # 类别编号

        s = time.time()

        # 源数据转tensor
        vid = fluid.dygraph.to_variable(vid)
        cls = fluid.dygraph.to_variable(cls)
        cls = reshape(cls, shape=[-1, 1])
        # forward
        out, acc = model(vid, cls)
        accs.append(acc)
        # loss
        loss = fluid.layers.cross_entropy(out, cls)
        avg_loss = mean(loss)
        # print('loss: ', float(avg_loss))
        e = time.time()
        print(' batch', batch_id, ' time', (e - s), ' loss', float(avg_loss), ' acc', acc)

        with open(os.path.join(log_path, 'log.json'), 'w') as out:
            json.dump(log, out)
print('total acc:', np.mean(accs))

# lr_sched.step()