# Reproduce-Article-Representation-Flow-for-Action-Recognition-with-PaddlePaddle
CVPR2019论文representation flow for action recognition的PaddlePaddle复现
原论文地址：https://arxiv.org/pdf/1810.01455
原论文Github源码地址：https://github.com/piergiaj/representation-flow-cvpr19

# 数据集

HMDB51数据集，split1划分
数据集名称为HMDB: a large human motion database

# 文件结构

| 文件 | 功能 |
| -------- | -------- |
|avi2jpg.py|avi视频中提取jpg图像帧|
|jpg2pkl.py|按split1划分数据集|
|train_model.py|模型训练程序|
|infer.py|模型验证程序|
|hmdb_dataset.py|数据读取器|
|flow_2d_resnets.py|ResNet50主干网络结构|
|rep_flow_2d_layer.py|光流表示层网络结构|

# 运行流程

## 数据集处理
### 视频提取jpg帧
avi2jpg.py
### 划分数据集
jpg2pkl.py

## 模型训练
train_model.py -save_dir xxx

## 模型验证
infer.py -pretrain xxx

# 原运行环境
百度AIStudio，单V100 GPU
