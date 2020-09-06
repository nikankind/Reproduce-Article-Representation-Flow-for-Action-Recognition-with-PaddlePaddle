import os
import numpy as np
import pickle

import cv2
import sys
import glob
from multiprocessing import Pool
from natsort import natsorted


label_dic = np.load('label_dir.npy', allow_pickle=True).item()
print(label_dic)

split_dir='/media/nk/HGST_SAS_8TB/AI/Dataset/hmdb51_org/'
source_dir = '/media/nk/HGST_SAS_8TB/AI/Dataset/hmdb_dtst/'
target_train_dir = source_dir+'train/'
target_test_dir = source_dir+'test/'


# target_val_dir = source_dir+'val/'
if not os.path.exists(target_train_dir):
    os.mkdir(target_train_dir)
if not os.path.exists(target_test_dir):
    os.mkdir(target_test_dir)
#if not os.path.exists(target_val_dir):
#    os.mkdir(target_val_dir)

with open(split_dir+'train.txt') as f:
    train_arr=f.readlines()
with open(split_dir+'test.txt') as f:
    test_arr=f.readlines()


for itm in train_arr:
    vid_fl, cls = itm.split(' ')
    cls = cls.strip()
    vid = vid_fl[0:-4]
    frame = os.listdir(os.path.join(source_dir,cls+'_jpg',vid))
    frame = natsorted(frame)
    frame = [os.path.join(source_dir,cls+'_jpg', vid, itm) for itm in frame]
    output_pkl = os.path.join(target_train_dir, vid + '.pkl')
    with open(output_pkl, 'wb') as f:
        pickle.dump((vid, label_dic[cls], frame), f, -1)

for itm in test_arr:
    vid_fl, cls = itm.split(' ')
    cls = cls.strip()
    vid=vid_fl[0:-4]
    frame=os.listdir(os.path.join(source_dir,cls+'_jpg',vid))
    frame = [os.path.join(source_dir, cls + '_jpg', vid, itm) for itm in frame]
    output_pkl = os.path.join(target_test_dir, vid + '.pkl')
    with open(output_pkl,'wb') as f:
        pickle.dump((vid,label_dic[cls],frame),f,-1)
