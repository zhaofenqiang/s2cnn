# pylint: disable=E1101,R,C
import torch.nn as nn
import torch
import torch.nn.functional as F

from s2cnn import s2_equatorial_grid, S2Convolution, so3_equatorial_grid, SO3Convolution, so3_integrate

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = [3, 25, 25, 25, 36]
        self.bandwidths = [32, 32, 32, 32, 32]

		# S2 layer
        grid = s2_equatorial_grid(max_beta=0, n_alpha=2 * self.bandwidths[0], n_beta=1)
        self.s2_conv = S2Convolution(self.features[0], self.features[1], self.bandwidths[0], self.bandwidths[1], grid)
        self.bn1 = nn.BatchNorm3d(self.features[1], affine=True)
        self.relu1 = nn.ReLU()

        # SO3 layers
        b_in = 32
        b_out = 32
        grid3 = so3_equatorial_grid(max_beta=0, max_gamma=0, n_alpha=2 * 32, n_beta=1, n_gamma=1)
        self.so3_layer1 = SO3Convolution(self.features[1], self.features[2], b_in, b_out, grid3)
        self.bn2 = nn.BatchNorm3d(self.features[1] + self.features[2], affine=True)
        self.relu2 = nn.ReLU()
        self.so3_layer2 = SO3Convolution(self.features[1] + self.features[2], self.features[3], b_in, b_out, grid3)
        self.bn3 = nn.BatchNorm3d(self.features[1] + self.features[2] + self.features[3], affine=True)
        self.relu3 = nn.ReLU()
        self.so3_layer3 = SO3Convolution(self.features[1] + self.features[2] + self.features[3], self.features[4], b_in, b_out, grid3)
        self.bn4 = nn.BatchNorm3d(self.features[4], affine=True)
        self.relu4 = nn.ReLU()

    def _make_SO3_layer(self, l):
        sequence = []
        nfeature_in = self.features[l] * l
        nfeature_out = self.features[l + 1]
        b_in = self.bandwidths[l]
        b_out = self.bandwidths[l + 1]

        
        sequence.append(SO3Convolution(nfeature_in, nfeature_out, b_in, b_out, grid))
		#sequence.append(nn.Dropout(p=0.5))

        return nn.Sequential(*sequence)

    def forward(self, x):  # pylint: disable=W0221

#        x = self.s2_conv(x)  
#        x = self.bn1(x)
#        x1 = self.relu1(x)
#        x = self.so3_layer1(x1)
#        x = x + x1
#        x2 = self.relu2(x)
#        x = self.so3_layer2(x2)
#        x = x + x2
#        x = self.relu3(x)
#
#        x = torch.mean(x, -1)  
 
        x1 = self.s2_conv(x)  # 4*20*64*64*64
        x = self.bn1(x1)
        x = self.relu1(x)
        x2 = self.so3_layer1(x)  # 4*20*64*64*64
        x = torch.cat((x1, x2), 1)  # 4*40*64*64*64
        x = self.bn2(x)
        x = self.relu2(x)
        x3 = self.so3_layer2(x)   # 4*20*64*64*64
        x = torch.cat((x1, x2, x3), 1)  # 4*60*64*64*64
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.so3_layer3(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = torch.mean(x, -1)  

        return x
