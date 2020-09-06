# import torch
# import torch.utils.data as data_utl

import numpy as np
import random

import os

from PIL import Image
from io import BytesIO
import pickle

import paddle
import functools
import logging

logger = logging.getLogger(__name__)

class HMDB():

    def __init__(self,
                 hmdb_pth='/media/nk/HGST_SAS_8TB/AI/Dataset/hmdb_dtst/',
                 pkl_pth='train/',
                 mode='rgb',
                 length=16,
                 model='2d',
                 random=True,
                 num_threads=6,
                 buf_size=1024,
                 batch_size=8
                 # c2i={}
                 ):  #split_file
        # split_file为文本文件，格式为：文件名 类别名


        cid = 0  # class-id 类别对应的编号
        # self.data为文件名-类编号列表
        self.data = []
        self.model = model
        self.size = 112

        self.hmdb_pth = hmdb_pth
        self.pkl_pth = os.path.join(hmdb_pth, pkl_pth)
        self.mode = mode
        self.length = length
        self.random = random
        self.num_reader_threads = num_threads
        self.buf_size = buf_size
        self.batch_size=batch_size

        for fil in os.listdir(self.pkl_pth):
            _, label, frames = pickle.load(open(os.path.join(self.pkl_pth, fil), 'rb'), encoding='bytes')
            self.data.append([frames, label])

    # 读一幅图
    def imageloader(self, buf):
        if isinstance(buf, str):
            img = Image.open(buf)
        else:
            img = Image.open(BytesIO(buf))

        return img.convert('RGB')  # H,W,C

    def video_loader(self, index):
        frames, cls = self.data[index]  # 图像文件list, 类别编号
        df = []
        # print(frames[0])
        for frm in frames:
            df.append(np.array(self.imageloader(frm)))  # [[H,W,C],...]
        df = np.array(df)
        t, h, w, _ = np.shape(df)

        # 抽帧
        if t > self.length * 2:
            st = random.randint(0, t - self.length * 2) if self.random else 0
            df = df[st:st + self.length * 2]
        elif t < self.length * 2 and t >= self.length:
            df = np.concatenate((df, df[::-1, :, :, :]))
            st = random.randint(0, t * 2 - self.length * 2) if self.random else 0
            df = df[st:st + self.length * 2]
        elif t < self.length:  # 对过短的视频返回空是否合适存疑
            print('video too short', frames)
            return None, None

        # 目标尺寸为源尺寸一半
        w = w // 2
        h = h // 2

        # center crop
        if not self.random:
            i = int(round((h - self.size) / 2.))
            j = int(round((w - self.size) / 2.))
            # df = np.reshape(df, newshape=(self.length*2, h*2, w*2, 3))[::2,::2,::2,:][:, i:-i, j:-j, :]
            if df.shape[1]>=2*self.size and df.shape[2]>=2*self.size:
                df = df[::2, ::2, ::2, :]
            else:
                df=df[::2,:,:,:]
            df = df[:, i:-i, j:-j, :]
        else:
            th = self.size  # 截取框高度
            tw = self.size  # 截取框宽度
            i = random.randint(0, h - th) if h > th else 0  # 截取框左上角y，随机
            j = random.randint(0, w - tw) if w > tw else 0  # 截取框左上角x，随机
            # df格式：(T,H,W,Channel)
            # 隔帧隔行隔列抽取并随机裁剪
            # df = np.reshape(df, newshape=(self.length*2, h*2, w*2, 3))[::2,::2,::2,:][:, i:i+th, j:j+tw, :]
            if df.shape[1]>=2*self.size and df.shape[2]>=2*self.size:
                df = df[::2, ::2, ::2, :]
            else:
                df=df[::2,:,:,:]
            df = df[:, i:i + th, j:j + tw, :]

        if self.mode == 'flow':
            # only take the 2 channels corresponding to flow (x,y)
            df = df[:, :, :, 1:]
            if self.model == '2d':
                # this should be redone...
                # stack 10 along channel axis
                df = np.asarray([df[:10], df[2:12], df[4:14]])  # gives 3x10xHxWx2
                df = df.transpose(0, 1, 4, 2, 3).reshape(3, 20, self.size, self.size).transpose(0, 2, 3, 1)

        df = 1 - 2 * (df.astype(np.float32) / 255)  # 一个trick, df在[-1,1]内

        if df.shape!=(16,112,112,3):
            print(df.shape,' ',self.data[index][0][0],'\n')
        # 2d -> return TxCxHxW
        return df.transpose([0, 3, 1, 2]), cls  # 0 3 1 2
        # 3d -> return CxTxHxW
        #return df.transpose([3, 0, 1, 2]), cls

    def create_reader(self):
        _reader = self._reader_creator(shuffle=self.random,
                                       num_threads=self.num_reader_threads,
                                       buf_size=self.buf_size)
        def _batch_reader():
            batch_out = []
            for imgs, label in _reader():
                if imgs is None:
                    continue
                batch_out.append((imgs, label))
                if len(batch_out) == self.batch_size:
                    yield batch_out
                    batch_out = []
        return _batch_reader

    def _reader_creator(self,
                        shuffle=False,
                        num_threads=1,
                        buf_size=1024):
        def reader():
            indexes = list(range(0, len(self.data)))
            if shuffle:
                random.shuffle(indexes)
            with open('a.pkl', 'wb') as f:
                pickle.dump(indexes, f, -1)
            for index in indexes:
                yield index
        mapper = functools.partial(
            self.video_loader

            )
        return paddle.reader.xmap_readers(mapper, reader, num_threads, buf_size)

    def __len__(self):
        return len(self.data)

    '''
    def imgs_transform(imgs, label, mode, seg_num, seglen, short_size,
                       target_size, img_mean, img_std):
        imgs = group_scale(imgs, short_size)

        if mode == 'train':
            if self.name == "TSM":
                imgs = group_multi_scale_crop(imgs, short_size)
            imgs = group_random_crop(imgs, target_size)
            imgs = group_random_flip(imgs)
            # 添加数据增强部分，提升分类精度
        else:
            imgs = group_center_crop(imgs, target_size)

        np_imgs = (np.array(imgs[0]).astype('float32').transpose(
            (2, 0, 1))).reshape(1, 3, target_size, target_size) / 255
        for i in range(len(imgs) - 1):
            img = (np.array(imgs[i + 1]).astype('float32').transpose(
                (2, 0, 1))).reshape(1, 3, target_size, target_size) / 255
            np_imgs = np.concatenate((np_imgs, img))
        imgs = np_imgs
        #imgs -= img_mean
        #imgs /= img_std
        imgs = np.reshape(imgs, (seg_num, seglen * 3, target_size, target_size))
        return imgs, label
    '''

if __name__ == '__main__':
    DS = HMDB
    dataseta = DS('data/hmdb/split1_train.txt', '/ssd/hmdb/', model='2d', mode='flow', length=16)
    dataset = DS('data/hmdb/split1_test.txt', '/ssd/hmdb/', model='2d', mode='rgb', length=16, c2i=dataseta.class_to_id)

    for i in range(len(dataseta)):
        print(dataseta[i][0].shape)
