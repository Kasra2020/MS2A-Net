# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 13:44:08 2022

@author: Kasra Rafiezadeh Shahi
"""
# =============================================================================
# Adding the required libraries
# =============================================================================



from __future__ import absolute_import, division, print_function, unicode_literals

import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import _supervised
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn import preprocessing


import torch
import torch.nn as nn



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Clustering Accuracy
# =============================================================================
def clustering_accuracy(labels_true, labels_pred):
    labels_true, labels_pred = _supervised.check_clusterings(labels_true, labels_pred)
    value = _supervised.contingency_matrix(labels_true, labels_pred)
    [r, c] = linear_sum_assignment(-value)
    return value[r, c].sum() / len(labels_true)


# =============================================================================
# Reading the input dataset
# =============================================================================


X = sio.loadmat('Trento.mat')['HSI']
[m,n,l] = X.shape
X = np.reshape(X,(X.shape[0]*X.shape[1],X.shape[2]))
min_max_scaler = preprocessing.MinMaxScaler()

y = sio.loadmat('Trento.mat')['GT']
y = np.reshape(y,(y.shape[0]*y.shape[1],-1))
y_test = y.reshape((m*n))
X = min_max_scaler.fit_transform(X)
X = np.float32(X)
ind = np.nonzero(y)


atrous_features = 12

no_features = np.int(3*atrous_features)


class MS2A_net(nn.Module):
    def __init__(self):
        super(MS2A_net, self).__init__()
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(l, atrous_features, 5, 1, 2, 1),
            nn.BatchNorm2d(atrous_features),
            nn.ReLU(),
        )
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(l, atrous_features, 5, 1, 4, 2),
            nn.BatchNorm2d(atrous_features),
            nn.ReLU(),
        )
        self.conv4_3 = nn.Sequential(
            nn.Conv2d(l, atrous_features, 5, 1, 8, 4),
            nn.BatchNorm2d(atrous_features),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(l, 3, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )
        self.conv5_1 = nn.Sequential(
            nn.Conv2d(no_features, 3, 1, 2),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )
        self.dconv4 = nn.Sequential(
            nn.Conv2d(6, l, 3, 1, 1),
            nn.BatchNorm2d(l),
            nn.ReLU(),
        )
    def forward(self, x):
        
        # =============================================================================
        #       Spectral-associate stream
        # =============================================================================
        x_in1_1 = torch.reshape(x,(x.shape[1],x.shape[2]*x.shape[3]))
        x_in2 = self.conv5(x)
        x_in2 = torch.reshape(x_in2,(x_in2.shape[1],x.shape[2]*x.shape[3]))
        sft = nn.Softmax(dim=0)
        x_in2 = sft(x_in2)
        x_in3 = sft(x_in1_1)
        x_in2 = torch.transpose(x_in2,0,1)
        x_in4 = torch.matmul(x_in3, x_in2)
        x_in4 = torch.transpose(x_in4,0,1)
        x_input = torch.matmul(x_in4, x_in1_1)
        x_input = torch.reshape(x_input,(x.shape[0],x_input.shape[0],x.shape[2],x.shape[3]))

        # =============================================================================
        #        Multi-scale spatial stream    
        # =============================================================================
        x_4_2 = self.conv4_1(x)
        x_4_3 = self.conv4_2(x)
        x_4_4 = self.conv4_3(x)
        x = torch.cat((x_4_2, x_4_3, x_4_4), dim = 1)
        
        # =============================================================================
        #        Decoding phase
        # =============================================================================
        
        code = x
        x = self.conv5_1(x)
        x_de_1 = torch.cat((x_input,x),dim=1)
        x = self.dconv4(x_de_1)
        x = x.view(x.size(0), -1)
        code = code.view(code.size(1), -1)
        return x, code
    

# =============================================================================
# Setup the hyperparameters
# =============================================================================
N_cluster = 6
LR = 0.001
m2sa = MS2A_net()
m2sa.cuda()
Coef_mean = 0.1
Iter = 800
thr = 100
print(m2sa)
optimizer_m2sa = torch.optim.Adam(m2sa.parameters(), lr=LR)
loss_func = nn.MSELoss()
#=============================================================================


# =============================================================================
# MS2A-Net main architechture
# =============================================================================
start_time = time.time()
tmpt_org = X.transpose()
tmpt_mean = np.mean(X, axis=1)
tmpt_mean = tmpt_mean.transpose()
tmpt_loss = tmpt_org.reshape((1,m*n*l))
tmpt_S = tmpt_org.reshape((1,l,m,n))
Spatial_Data = torch.from_numpy(tmpt_S)
loss_ls = []

for i in range(Iter):
    Spat = Spatial_Data
    output_m2sa, code_m2sa = m2sa(Spat.cuda())
    code_mean = torch.mean(code_m2sa, dim=0)
    loss_cae = loss_func(output_m2sa, torch.from_numpy(tmpt_loss).cuda())
    loss_mean = loss_func(code_mean, torch.from_numpy(tmpt_mean).cuda())
    loss = loss_cae + (Coef_mean*loss_mean)
    optimizer_m2sa.zero_grad()
    loss.backward()
    optimizer_m2sa.step()
    loss_ls.append(loss)
    print('Iteration: ', i, '| Total loss: %.4f' % loss.data.cpu().numpy())
    if loss.data.cpu().numpy() < thr:
        torch.save(m2sa.state_dict(), 'Transformer_clustering/net_params_m2saRecons.pkl')
        thr = loss.data.cpu().numpy()
    #=============================================================================
    

    
m2sa_1 = MS2A_net().cuda()
model_dict = m2sa_1.state_dict()
pretrained_dict = torch.load('Transformer_clustering/net_params_m2saRecons.pkl')
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)  
m2sa_1.load_state_dict(model_dict)



Z_2 = m2sa_1.conv4_1(torch.from_numpy(tmpt_S).cuda())
Z_3 = m2sa_1.conv4_2(torch.from_numpy(tmpt_S).cuda())
Z_4 = m2sa_1.conv4_3(torch.from_numpy(tmpt_S).cuda())
Z = torch.cat((Z_2,Z_3,Z_4), dim = 1)
Z = Z.detach().cpu().numpy()
Z = Z.reshape((no_features,m*n))
Z = Z.transpose()




# =============================================================================
# Clustering step
# =============================================================================


KM = KMeans(n_clusters=N_cluster, random_state=0)
CS = KM.fit(Z)
CSmap = np.zeros((m*n))
CSmap = CS.labels_ + 1
CA = clustering_accuracy(y_test[ind[0]], CSmap[ind[0]])
NMI = normalized_mutual_info_score(y_test[ind[0]], CSmap[ind[0]])
ARI = adjusted_rand_score(y_test[ind[0]], CSmap[ind[0]])
print('CA:\t'+np.str(CA)+'\n'+'NMI:\t'+np.str(NMI)+'\n'+'ARI:\t'+np.str(ARI))
CSmap = CSmap.reshape((m,n))

# # =============================================================================
# # Visualization
# # =============================================================================


fig, (ax1, ax2) = plt.subplots(1,2)
fig.suptitle('Clustering result')
ax1.imshow(y_test.reshape((m,n)))
ax1.set_title('GT')
ax1.set_yticklabels([])
ax1.set_xticklabels([])
ax2.imshow(CSmap)
ax2.set_title('Original')
ax2.set_yticklabels([])
ax2.set_xticklabels([])
end_time = time.time()


P_time = end_time - start_time
print(P_time)

# =============================================================================
# Saving the generated clustering map as a .mat file    
# =============================================================================

# sio.savemat('Clustering_map.mat', {'CSmap':CSmap}) 
    
    
    
    
    
    