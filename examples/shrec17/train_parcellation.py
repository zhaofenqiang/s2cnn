# pylint: disable=E1101,R,C,W1202
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 11:55:31 2018

@author: zfq
"""

import torch
import torch.nn.functional as F
import torchvision
import torch.nn as nn

import glob
import os
import scipy.io as sio 
import shutil
import time
import logging
import copy
import types
import importlib.machinery
import numpy as np
from tensorboardX import SummaryWriter

writer = SummaryWriter('log/dataset64_1fold_4layer_with_dp')

batch_size = 4
learning_rate = 0.5  
num_workers = 4
wd = 0.0001

class SphereSurf(torch.utils.data.Dataset):

    def __init__(self, root, transform=None):
        self.root = root
        self.files = sorted(glob.glob(os.path.join(self.root, '*.mat')))    
        self.transform = transform
        
    def __getitem__(self, index):
        file = self.files[index]
        img = sio.loadmat(file)
        img = img['data']
        img = np.transpose(img)
        img = np.reshape(img, (3,64,64)).astype(np.float32)
        label = sio.loadmat(file[:-4] + '.label')
        label = label['label']    
        label = np.squeeze(label)
        label = label - 1
        label = np.reshape(label, (64,64)).astype(np.long)
        return img, label

    def __len__(self):
        return len(self.files)

torch.backends.cudnn.benchmark = True

# Load the model
loader = importlib.machinery.SourceFileLoader('model', "parcellation_model.py")
mod = types.ModuleType(loader.name)
loader.exec_module(mod)

model = mod.Model()
model.cuda()

train_dataset = SphereSurf("/media/zfq/WinE/unc/zhengwang/dataset_64/train")
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
test_dataset = SphereSurf("/media/zfq/WinE/unc/zhengwang/dataset_64/val")
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
 
optimizer = torch.optim.SGD(model.parameters(), lr=0, momentum=0.9, weight_decay=wd)
criterion = nn.CrossEntropyLoss()

def train_step(data, target):
    model.train()
    data, target = data.cuda(), target.cuda()

    prediction = model(data)
    
    loss = criterion(prediction, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    correct = prediction.data.max(1)[1].eq(target.data).long().cpu().sum()

    return loss.item(), correct.item()

def get_learning_rate(epoch):
    limits = [60, 120]
    lrs = [1, 0.1, 0.01]
    assert len(lrs) == len(limits) + 1
    for lim, lr in zip(limits, lrs):
        if epoch < lim:
            return lr * learning_rate
    return lrs[-1] * learning_rate

def test_during_training():
    model.eval()
    total_correct = 0
    for batch_idx, (data, target) in enumerate(test_dataloader):
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            prediction = model(data)
        total_correct += prediction.data.max(1)[1].eq(target.data).long().cpu().sum() 

    acc = total_correct.item() / (batch_size * len(test_dataloader) * 64 * 64)    
    return acc

for epoch in range(300):
    lr = get_learning_rate(epoch)
    print("learning rate = {} and batch size = {}".format(lr, train_dataloader.batch_size))
    for p in optimizer.param_groups:
        p['lr'] = lr

    total_loss = 0
    total_correct = 0
    
#    dataiter = iter(train_dataloader)
#    data, target = dataiter.next()
    
    for batch_idx, (data, target) in enumerate(train_dataloader):
        loss, correct = train_step(data, target)

        total_loss += loss
        total_correct += correct

        print("[{}:{}/{}] LOSS={:.2} <LOSS>={:.2} ACC={:.4} <ACC>={:.4}".format(
            epoch, batch_idx, len(train_dataloader),
            loss, total_loss / (batch_idx + 1),
            correct / (target.size()[0] * target.size()[1] * target.size()[2]), 
            total_correct / (target.size()[0] * target.size()[1] * target.size()[2] * (batch_idx+1))))

    writer.add_scalar('Train/Loss', total_loss / len(train_dataloader), epoch)
    test_acc = test_during_training()
    print("Test ACC= ", test_acc)
    writer.add_scalars('data/Acc', {'train': total_correct / (batch_size * 64 * 64 * len(train_dataloader)),
                                   'val': test_acc}, epoch)
    if epoch % 10 == 0 :
        torch.save(model.state_dict(), os.path.join("state.pkl"))
    

##%%
#root = "/media/zfq/WinE/unc/zhengwang/dataset1/train"
#files = glob.glob(os.path.join(root, '*.mat'))
#labels = sorted(glob.glob(os.path.join(root, '*.label')))    
#file = files[0]
#img = sio.loadmat(file)
#img = img['data']
#img = np.transpose(img)
#img = np.reshape(img, (3,128,128)).astype(np.float32)
#label = sio.loadmat(file[:-4] + '.label')
#label = label['label']   
#label = np.squeeze(label)
##%%
#import matplotlib.pyplot as plt  
#import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
#
##a = im[0,:,:]
##fig = plt.figure()  
##ax = fig.add_subplot(111)  
##ax.imshow(a) 
#
#def rotmat(a, b, c, hom_coord=False):  # apply to mesh using mesh.apply_transform(rotmat(a,b,c, True))
#    """
#    Create a rotation matrix with an optional fourth homogeneous coordinate
#
#    :param a, b, c: ZYZ-Euler angles
#    """
#    def z(a):
#        return np.array([[np.cos(a), np.sin(a), 0, 0],
#                         [-np.sin(a), np.cos(a), 0, 0],
#                         [0, 0, 1, 0],
#                         [0, 0, 0, 1]])
#
#    def y(a):
#        return np.array([[np.cos(a), 0, np.sin(a), 0],
#                         [0, 1, 0, 0],
#                         [-np.sin(a), 0, np.cos(a), 0],
#                         [0, 0, 0, 1]])
#
#    r = z(a).dot(y(b)).dot(z(c))  # pylint: disable=E1101
#    if hom_coord:
#        return r
#    else:
#        return r[:3, :3]
#
#
#def make_sgrid(b, alpha, beta, gamma):
#    from lie_learn.spaces import S2
#
#    theta, phi = S2.meshgrid(b=b, grid_type='SOFT')
#    sgrid = S2.change_coordinates(np.c_[theta[..., None], phi[..., None]], p_from='S', p_to='C')
#    sgrid = sgrid.reshape((-1, 3))
#
#    R = rotmat(alpha, beta, gamma, hom_coord=False)
#    sgrid = np.einsum('ij,nj->ni', R, sgrid)
#
#    return sgrid
#
#
#sgrid = make_sgrid(64, alpha=0, beta=0, gamma=0)
#
#x, y, z = sgrid[:,0], sgrid[:,1], sgrid[:,2]
#ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
#ax.scatter(x, y, z, s=1, c=img[:,2])  # 绘制数据点    
#ax.set_zlabel('Z')  # 坐标轴
#ax.set_ylabel('Y')
#ax.set_xlabel('X')
#plt.show()
