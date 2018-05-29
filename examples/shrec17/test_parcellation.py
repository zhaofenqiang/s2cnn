# pylint: disable=E1101,R,C
import os
import numpy as np
import torch
import types
import importlib.machinery
import glob
import torchvision
import scipy.io as sio 

batch_size = 4
num_workers = 1

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

# Load the model
loader = importlib.machinery.SourceFileLoader('model', "parcellation_model.py")
mod = types.ModuleType(loader.name)
loader.exec_module(mod)

model = mod.Model()
model.cuda()
model.load_state_dict(torch.load("state.pkl"))

test_dataset = SphereSurf("/media/zfq/WinE/unc/zhengwang/dataset2_64/val")
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
 
total_correct = 0 
model.eval()
for batch_idx, (data, target) in enumerate(test_dataloader):   
    data, target = data.cuda(), target.cuda()
    with torch.no_grad():
        prediction = model(data)

    total_correct += prediction.data.max(1)[1].eq(target.data).long().cpu().sum()
    
#    dataiter = iter(test_dataset)
#    data, target = dataiter.next()

acc = total_correct.item() / (batch_size * len(test_dataloader) * 64 * 64)    
print("Test ACC= ", acc)
