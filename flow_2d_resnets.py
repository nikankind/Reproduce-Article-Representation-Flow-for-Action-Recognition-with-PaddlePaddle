import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torch.utils.model_zoo as model_zoo

from rep_flow_2d_layer import FlowLayer
import paddle
from paddle import fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear, Dropout
from paddle.fluid.layers import concat, pad, reshape, reduce_mean, accuracy

################
#
# Modified https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# Adds support for B x T x C x H x W video data
#
################


__all__ = ['ResNet', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# Conv + BN + Activator
class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 padding=0,
                 groups=1,
                 act=None):
        super(ConvBNLayer, self).__init__(name_scope)

        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,  # (filter_size - 1) // 2,  # Conv后tensor尺寸不变
            groups=groups,
            act=None,
            bias_attr=False)
        self._batch_norm = BatchNorm(num_filters, act=act)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)

        return y

class Bottleneck(fluid.dygraph.Layer):
    expansion = 4

    def __init__(self, name_scope, num_channels, num_filters, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = ConvBNLayer(self.full_name(), num_channels=num_channels, num_filters=num_filters, filter_size=1, padding=0, act='relu')
        self.conv2 = ConvBNLayer(self.full_name(), num_channels=num_filters, num_filters=num_filters, filter_size=3, padding=1, stride=stride, act='relu')
        self.conv3 = ConvBNLayer(self.full_name(), num_channels=num_filters, num_filters=num_filters * self.expansion, padding=0, filter_size=1, act=None)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = fluid.layers.elementwise_add(x=residual, y=out, act='relu')

        return out


class ResNet(fluid.dygraph.Layer):
    # block是Bottleneck
    def __init__(self, name_scope, block, layers, inp=3, num_classes=51, input_size=112, dropout=0.5, n_iter=20, learnable=[0, 1, 1, 1, 1]):
        self.inplanes = 64
        self.inp = inp
        super(ResNet, self).__init__(name_scope)

        ###新增：光流表示层FCF
        self.flow_cmp = Conv2D(128 * block.expansion, 32, filter_size=1, stride=1, padding=0, bias_attr=False)
        self.flow_layer = FlowLayer(channels=32, n_iter=n_iter, params=learnable)  # 光流表示层
        self.flow_conv = Conv2D(64, 64, filter_size=3, stride=1, padding=1, bias_attr=False)

        # Flow-of-flow
        self.flow_cmp2 = Conv2D(64, 32, filter_size=1, stride=1, padding=0, bias_attr=False)
        self.flow_layer2 = FlowLayer(channels=32, n_iter=n_iter, params=learnable)  # 光流表示层
        self.flow_conv2 = Conv2D(64, 128 * block.expansion, filter_size=3, stride=1, padding=1, bias_attr=False)
        self.bnf = BatchNorm(128 * block.expansion, act='relu')

        ###

        self.conv1 = ConvBNLayer(self.full_name(),num_channels=inp, num_filters=64, filter_size=7, stride=2, padding=3)
        self.maxpool = Pool2D(pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')
        # resnet50 layers=[3,4,6,3]
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # probably need to adjust this based on input spatial size
        size = int(math.ceil(input_size/32))
        self.avgpool = Pool2D(pool_size=size, pool_stride=1, pool_type='avg')
        self.dropout = Dropout(p=dropout)
        self.fc = Linear(input_dim=512 * block.expansion, output_dim=num_classes, bias_attr=False, act='softmax')

        for m in self.sublayers():
            if isinstance(m, Conv2D):
                # 实验性方法，kaiming初始化
                m = Conv2D(num_channels=m._num_channels, num_filters=m._num_filters, filter_size=m._filter_size,
                           stride=m._stride, padding=m._padding, groups=m._groups, act=m._act, bias_attr=m._bias_attr,
                           param_attr=fluid.initializer.MSRAInitializer(uniform=False))
        '''    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        '''

    def _make_layer(self, block, planes, blocks, stride=1):
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = ConvBNLayer(self.full_name(), num_channels=self.inplanes, num_filters=planes*block.expansion, filter_size=1, stride=stride, act=None)
        else:
            downsample = None
        layers = []
        layers.append(block(self.full_name(), num_channels=self.inplanes, num_filters=planes, stride=stride, downsample=downsample))  # layers[0]
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.full_name(), num_channels=self.inplanes, num_filters=planes))
        # 各blocks串联起来
        return fluid.dygraph.Sequential(*layers)

    def forward(self, x, cls=None):
        # x is BxTxCxHxW 注意与2p1d网络输入格式不同
        # spatio-temporal video data
        b,t,c,h,w = x.shape
        # need to view it is B*TxCxHxW for 2D CNN
        # important to keep batch and time axis next to
        # eachother, so a simple view without tranposing is possible
        # 此处存疑，因为torch.dataloader作batch打包录入数据时，各类别是混起来的，而且同类视频间也不方便混起来的，因为要计算表示层光流
        x = reshape(x, shape=[b*t, c, h, w])
        
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)

        # 插入FCF层
        #b_t, c, h, w = x.size()
        # res = x  # F.avg_pool2d(x, (3, 1), 1, 0)  # x[:,:,1:-1].contiguous() F表示torch.nn.functional
        x = self.flow_cmp(x)
        x = self.flow_layer.norm_img(x)
        
        # 将batch中各视频分开后再送Representation Flow
        # x=x.view(b,t,c,h,w)

        # handle time
        # _, c, t, h, w = x.size()
        # t = t - 1
        # compute flow for 0,1,...,T-1
        #        and       1,2,...,T
        b_t, c, h, w = x.shape
        x = reshape(x,shape=[b,-1,c,h,w])
        t -= 1  # Representation Flow操作后，t少一帧
        u, v = self.flow_layer(reshape(x[:,:-1], shape=[-1,c,h,w]), reshape(x[:,1:], shape=[-1,c,h,w]))
        # u, v = self.flow_layer(pad(x[:-1], [0, 1, 0, 0, 0, 0, 0, 0]), pad(x[1:], [1, 0, 0, 0, 0, 0, 0, 0]))
        x = concat([u, v], axis=1)

        # x = x.view(b, t, c * 2, h, w).permute(0, 2, 1, 3, 4).contiguous()
        x = self.flow_conv(x)

        # Flow-of-flow
        # _, ci, t, h, w = x.size()
        x = self.flow_cmp2(x)
        x = self.flow_layer.norm_img(x)
        # handle time
        # _, c, t, h, w = x.size()
        # t = t - 1
        # compute flow for 0,1,...,T-1
        #        and       1,2,...,T
        b_t, c, h, w = x.shape
        x = reshape(x, shape=[b, -1, c, h, w])
        t -= 1  # Representation Flow操作后，t少一帧
        u, v = self.flow_layer2(reshape(x[:,:-1], shape=[-1,c,h,w]), reshape(x[:,1:], shape=[-1,c,h,w]))
        x = concat([u, v], axis=1)
        # x = x.view(b, t, c * 2, h, w).permute(0, 2, 1, 3, 4).contiguous()
        x = self.flow_conv2(x)
        x = self.bnf(x)
        # x = x + res
        # x = self.relu(x)

        #

        x = self.layer3(x)
        x = self.layer4(x)

        #print(x.size())
        x = self.avgpool(x)

        x = reshape(x, shape=[x.shape[0], -1])
        x = self.dropout(x)

        # currently making dense, per-frame predictions
        x = self.fc(x)

        # so view as BxTxClass
        x = reshape(x, shape=[b, t, -1])
        # mean-pool over time
        x = reduce_mean(x, dim=1)  # temporal维度合并

        # return BxClass prediction
        if cls is not None:
            acc = float(accuracy(input=x, label=cls))
            return x, acc
        else:
            return x

    def load_state_dict(self, state_dict, strict=True):
        # ignore fc layer
        state_dict = {k:v for k,v in state_dict.items() if 'fc' not in k}
        md = self.state_dict()
        md.update(state_dict)
        # convert to flow representation
        if self.inp != 3:
            for k,v in md.items():
                if k == 'conv1.weight':
                    if isinstance(v, nn.Parameter):
                        v = v.data
                    # change image CNN to 20-channel flow by averaing RGB channels and repeating 20 times
                    v = torch.mean(v, dim=1).unsqueeze(1).repeat(1, self.inp, 1, 1)
                    md[k] = v
        
        super(ResNet, self).load_state_dict(md, strict)


def resnet50(pretrained=False, mode='rgb', **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if mode == 'flow':
        model = ResNet('resnet50', Bottleneck, [3, 4, 6, 3], inp=20, **kwargs)
    else:
        model = ResNet('resnet50', Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet('resnet101', Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet('resnet152', Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model



if __name__ == '__main__':
    # test resnet 50
    import torch
    d = torch.device('cuda')
    net = resnet50(pretrained=True, mode='flow')
    net.to(d)

    vid = torch.rand((4,32,20,112,112)).to(d)

    print(net(vid).size())
