'''
import torch
import torch.nn as nn
import torch.nn.functional as F
'''
import math
import numpy as np
import paddle
from paddle import fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear, Dropout
from paddle.fluid.initializer import NumpyArrayInitializer, ConstantInitializer
from paddle.fluid.layers import reduce_max, reduce_min, zeros_like, logical_and, logical_or, logical_not, cast, pad2d
from paddle.fluid.layers import elementwise_add, elementwise_sub, elementwise_mul, elementwise_div, sqrt


class FlowLayer(fluid.dygraph.Layer):
    w_param_attrs = fluid.ParamAttr

    # params为某参数是否可学习
    def __init__(self, channels=1, params=[0, 1, 1, 1, 1], n_iter=10):
        super(FlowLayer, self).__init__()
        self.n_iter = n_iter
        sobel = np.kron(np.resize(np.eye(channels), [channels, channels, 1, 1]),
                        np.array([[[[-0.5, 0, 0.5], [-1, 0, 1], [-0.5, 0, 0.5]]]]))  # Sobel矩阵
        wx = np.array([[[[-1, 1]]]]).repeat(channels, axis=0)
        wy = np.array([[[[-1], [1]]]]).repeat(channels, axis=0)
        if params[0]:
            self.conv_img_grad = Conv2D(num_channels=channels, num_filters=channels, filter_size=3,
                                        padding=1, stride=1, bias_attr=False,
                                        param_attr=fluid.ParamAttr(initializer=NumpyArrayInitializer(value=sobel))
                                        )
            self.conv_img_grad2 = Conv2D(num_channels=channels, num_filters=channels, filter_size=3,
                                         padding=1, stride=1, bias_attr=False,
                                         param_attr=fluid.ParamAttr(
                                             initializer=NumpyArrayInitializer(value=sobel.transpose([0, 1, 3, 2])))
                                         )
            self.conv_f_grad = Conv2D(num_channels=channels, num_filters=channels, filter_size=3,
                                      padding=1, stride=1, bias_attr=False,
                                      param_attr=fluid.ParamAttr(initializer=NumpyArrayInitializer(value=sobel))
                                      )
            self.conv_f_grad2 = Conv2D(num_channels=channels, num_filters=channels, filter_size=3,
                                       padding=1, stride=1, bias_attr=False,
                                       param_attr=fluid.ParamAttr(
                                           initializer=NumpyArrayInitializer(value=sobel.transpose([0, 1, 3, 2])))
                                       )

        else:
            self.conv_img_grad = Conv2D(num_channels=channels, num_filters=channels, filter_size=3,
                                        padding=1, stride=1, bias_attr=False,
                                        param_attr=fluid.ParamAttr(initializer=NumpyArrayInitializer(value=sobel),
                                                                   trainable=False)
                                        )
            self.conv_img_grad2 = Conv2D(num_channels=channels, num_filters=channels, filter_size=3,
                                         padding=1, stride=1, bias_attr=False,
                                         param_attr=fluid.ParamAttr(
                                             initializer=NumpyArrayInitializer(value=sobel.transpose([0, 1, 3, 2])),
                                             trainable=False)
                                         )
            self.conv_f_grad = Conv2D(num_channels=channels, num_filters=channels, filter_size=3,
                                      padding=1, stride=1, bias_attr=False,
                                      param_attr=fluid.ParamAttr(initializer=NumpyArrayInitializer(value=sobel),
                                                                 trainable=False)
                                      )
            self.conv_f_grad2 = Conv2D(num_channels=channels, num_filters=channels, filter_size=3,
                                       padding=1, stride=1, bias_attr=False,
                                       param_attr=fluid.ParamAttr(
                                           initializer=NumpyArrayInitializer(value=sobel.transpose([0, 1, 3, 2])),
                                           trainable=False)
                                       )

        if params[1]:

            self.conv_div = Conv2D(num_channels=channels, num_filters=channels, filter_size=(1, 2),
                                   padding=0, stride=1, bias_attr=False, groups=channels,
                                   param_attr=fluid.ParamAttr(initializer=NumpyArrayInitializer(value=wx))
                                   )
            self.conv_div2 = Conv2D(num_channels=channels, num_filters=channels, filter_size=(2, 1),
                                    padding=0, stride=1, bias_attr=False, groups=channels,
                                    param_attr=fluid.ParamAttr(initializer=NumpyArrayInitializer(value=wy))
                                    )

        else:

            self.conv_div = Conv2D(num_channels=channels, num_filters=channels, filter_size=(1, 2),
                                   padding=0, stride=1, bias_attr=False, groups=channels,
                                   param_attr=fluid.ParamAttr(initializer=NumpyArrayInitializer(value=wx),
                                                              trainable=False)
                                   )
            self.conv_div2 = Conv2D(num_channels=channels, num_filters=channels, filter_size=(2, 1),
                                    padding=0, stride=1, bias_attr=False, groups=channels,
                                    param_attr=fluid.ParamAttr(initializer=NumpyArrayInitializer(value=wy),
                                                               trainable=False)
                                    )

        self.channels = channels

        self.t = 0.3  # theta
        self.l = 0.15  # lambda
        self.a = 0.25  # tau

        if params[2]:
            self.t = fluid.layers.create_parameter(shape=[1], dtype='float32',
                                                   attr=fluid.ParamAttr(initializer=ConstantInitializer(value=self.t)))
        if params[3]:
            self.l = fluid.layers.create_parameter(shape=[1], dtype='float32',
                                                   attr=fluid.ParamAttr(initializer=ConstantInitializer(value=self.l)))
        if params[4]:
            self.a = fluid.layers.create_parameter(shape=[1], dtype='float32',
                                                   attr=fluid.ParamAttr(initializer=ConstantInitializer(value=self.a)))

    def norm_img(self, x):
        mx = reduce_max(x)
        mn = reduce_min(x)
        x =  (x - mn) / (mx - mn)  # 原为(mn-mx)
        return x

    def forward_grad(self, x):
        #grad_x = self.conv_f_grad(x)
        #grad_y = self.conv_f_grad2(x)
        grad_x = self.conv_f_grad(pad2d(x,(0,0,0,1)))
        grad_y = self.conv_f_grad2(pad2d(x,(0,1,0,0)))
        return pad2d(grad_x[:,:,:,:-1],(0,0,0,1)), pad2d(grad_y[:,:,:-1,:],(0,1,0,0))
        #return grad_x, grad_y

    def divergence(self, x, y):
        # tx = F.pad(x[:, :, :, :-1], (1, 0, 0, 0))
        # ty = F.pad(y[:, :, :-1, :], (0, 0, 1, 0))
        # grad_x = F.conv2d(F.pad(tx, (0, 1, 0, 0)), self.div, groups=self.channels)
        # grad_y = F.conv2d(F.pad(ty, (0, 0, 0, 1)), self.div2, groups=self.channels)
        tx = pad2d(x[:, :, :, :-1], (0, 0, 1, 0))  # 0,0,1,0
        ty = pad2d(y[:, :, :-1, :], (1, 0, 0, 0))  # 1,0,0,0
        grad_x = self.conv_div(pad2d(tx, (0, 0, 0, 1)))  # 0,0,0,1
        grad_y = self.conv_div2(pad2d(ty, (0, 1, 0, 0)))  # 0,1,0,0
        return grad_x + grad_y

    def forward(self, x, y):
        # x,y误差一帧
        u1 = zeros_like(x)
        u2 = zeros_like(x)
        l_t = self.l * self.t
        taut = self.a / self.t

        grad2_x = self.conv_img_grad(y)
        # grad2_x[:, :, :, 0] = 0.5 * (x[:, :, :, 1] - x[:, :, :, 0])
        # grad2_x[:, :, :, -1] = 0.5 * (x[:, :, :, -1] - x[:, :, :, -2])

        grad2_y = self.conv_img_grad2(y)
        # grad2_y[:, :, 0, :] = 0.5 * (x[:, :, 1, :] - x[:, :, 0, :])
        # grad2_y[:, :, -1, :] = 0.5 * (x[:, :, -1, :] - x[:, :, -2, :])

        p11 = zeros_like(x)
        p12 = zeros_like(x)
        p21 = zeros_like(x)
        p22 = zeros_like(x)

        gsqx = grad2_x ** 2
        gsqy = grad2_y ** 2
        grad = gsqx + gsqy + 1e-12

        rho_c = y - grad2_x * u1 - grad2_y * u2 - x

        for i in range(self.n_iter):
            rho = rho_c + grad2_x * u1 + grad2_y * u2 + 1e-12

            v1 = zeros_like(x)
            v2 = zeros_like(x)
            mask1 = rho < -l_t * grad
            mask2 = rho > l_t * grad
            mask3 = logical_and(logical_not(logical_or(mask1, mask2)), (grad > 1e-12))
            mask1 = cast(mask1, dtype='float32')
            mask2 = cast(mask2, dtype='float32')
            mask3 = cast(mask3, dtype='float32')
            mask1.stop_gradient = True
            mask2.stop_gradient = True
            mask3.stop_gradient = True

            # v1 = v1 + l_t * grad2_x * mask1 - l_t * grad2_x * mask2 - (rho / grad) * grad2_x * mask3
            # v2 = v2 + l_t * grad2_y * mask1 - l_t * grad2_y * mask2 - (rho / grad) * grad2_y * mask3
            v1 = elementwise_add(u1, elementwise_add(elementwise_mul(l_t * grad2_x, mask1), elementwise_add(elementwise_mul(-l_t * grad2_x, mask2),elementwise_mul(-elementwise_div(rho, grad), elementwise_mul(grad2_x, mask3)))))
            v2 = elementwise_add(u2, elementwise_add(elementwise_mul(l_t * grad2_y, mask1), elementwise_add(elementwise_mul(-l_t * grad2_y, mask2),elementwise_mul(-elementwise_div(rho, grad), elementwise_mul(grad2_y, mask3)))))

            del rho
            del mask1
            del mask2
            del mask3

            v1 += u1
            v2 += u2

            u1 = v1 + self.t * self.divergence(p11, p12)
            u2 = v2 + self.t * self.divergence(p21, p22)
            del v1
            del v2
            u1 = u1
            u2 = u2

            u1x, u1y = self.forward_grad(u1)
            u2x, u2y = self.forward_grad(u2)

            p11 = (p11 + taut * u1x) / (1. + taut * sqrt(u1x ** 2 + u1y ** 2 + 1e-12))
            p12 = (p12 + taut * u1y) / (1. + taut * sqrt(u1x ** 2 + u1y ** 2 + 1e-12))
            p21 = (p21 + taut * u2x) / (1. + taut * sqrt(u2x ** 2 + u2y ** 2 + 1e-12))
            p22 = (p22 + taut * u2y) / (1. + taut * sqrt(u2x ** 2 + u2y ** 2 + 1e-12))
            del u1x
            del u1y
            del u2x
            del u2y

        return u1, u2
