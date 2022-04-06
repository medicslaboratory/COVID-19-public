# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 12:51:05 2020

@author: gosek
"""

import time
import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

#Find re-weighting factor when correcting for class imbalance if the majority class loss is multiplied by (1-proportion of majority examples)
def find_cw(frac_arr):
    cw = []
    for c in frac_arr:
        cw.append(1/(2*c*(1-c)))
    return cw

class DenseNet121(nn.Module):
        """Model modified.
        The architecture of our model is the same as standard DenseNet121
        except the classifier layer which has an additional sigmoid function.
        """
        def __init__(self, out_size):
            super(DenseNet121, self).__init__()
            self.densenet121 = torchvision.models.densenet121(pretrained=True)
    #        num_ftrs = self.densenet121.classifier.in_features
            self.densenet121.classifier = nn.Sequential(
                nn.Linear(1024, out_size),
                nn.Sigmoid()
            )
    
        def forward(self, x):
            features = self.densenet121.features(x)
            out = nn.functional.relu(features)
            out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out_class = self.densenet121.classifier(out)
            return out_class

class Model():
    
    def __init__(self, n_classes, checkpoint, use_gpu):
        model = DenseNet121(n_classes).cuda()
        model = torch.nn.DataParallel(model).cuda()
        self.model = model
        self.use_gpu = use_gpu
        self.class_loss = torch.nn.BCELoss(reduce = False)
        if checkpoint != None:
            modelCheckpoint = torch.load(checkpoint)
            self.model.load_state_dict(modelCheckpoint['state_dict'])


    def train (self, trMaxEpoch, dataLoaderTrain, dataLoaderVal, weighting = True):
        
        timestampTime = time.strftime("%H%M%S")
        timestampDate = time.strftime("%d%m%Y")
        launchTimestamp = timestampDate + '-' + timestampTime
        
        #SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.AdamW(self.model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001)
        
        #TRAIN THE NETWORK
        lossMIN = 100000
        
        for epochID in range(0, trMaxEpoch):
            
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            
            batchs, losst, losse = self.epochTrain(optimizer, dataLoaderTrain, dataLoaderVal, trMaxEpoch, weighting)
            lossVal = self.epochVal(dataLoaderVal, weighting)


            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime
            
            lossMIN = lossVal
            torch.save({'epoch': epochID + 1, 'state_dict': self.model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, 'm-epoch'+str(epochID)+'-' + launchTimestamp + '.pth.tar')
            print ('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] loss= ' + str(lossVal))
        
        return batchs, losst, losse        
    #-------------------------------------------------------------------------------- 
       
    def epochTrain(self, optimizer, dataLoaderTrain, dataLoaderVal, epochMax, weighting):
        
        batch = []
        losstrain = []
        losseval = []
        
        self.model.train()

        for batchID, input in enumerate(dataLoaderTrain):
            
            var_class_target = input[1].cuda(non_blocking = True)
            bs, c, h, w = input[0].size()
            varInput = input[0].view(-1, c, h, w).cuda(non_blocking = True)

            varOutput = self.model(varInput)
        
            #Hard coded prevalence of disease in the CheXpert dataset
            if weighting:
                c_weights = torch.Tensor(find_cw([0.0483, 0.1208, 0.4725, 0.0411, 0.2338, 0.0661, 0.0270, 0.1493, 0.3857, 0.01576, 0.514])).cuda() 
                weights = torch.Tensor([0.0483, 0.1208, 0.4725, 0.0411, 0.2338, 0.0661, 0.0270, 0.1493, 0.3857, 0.01576, 0.514]).cuda() 
                weights_tensor = torch.where(torch.eq(var_class_target, 1), (var_class_target-weights), weights)
                lossvalue = (weights_tensor * self.class_loss(varOutput, var_class_target) * c_weights).mean() 
            
            else: 
                lossvalue = self.class_loss(varOutput, var_class_target).mean()
                       
            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()
            
            l = lossvalue.item()
            losstrain.append(l)
            
            if batchID%140==0:
                print(batchID//140, "% batches computed")
                #Fill three arrays to see the evolution of the loss

                batch.append(batchID)
                
                le = self.epochVal(dataLoaderVal, weighting).item()
                losseval.append(le)
                
                print(batchID)
                print(l)
                print(le)
                
        return batch, losstrain, losseval
    
    #-------------------------------------------------------------------------------- 
    
    def epochVal(self, dataLoaderVal, weighting):
        
        self.model.eval()
        
        lossVal = 0
        lossValNorm = 0

        with torch.no_grad():
            for i, input in enumerate(dataLoaderVal):
                
                var_class_target = input[1].cuda(non_blocking = True)
                bs, c, h, w = input[0].size()
                varInput = input[0].view(-1, c, h, w).cuda(non_blocking = True)
                varOutput = self.model(varInput)
                
                if weighting:
                    c_weights = torch.Tensor(find_cw([0.0483, 0.1208, 0.4725, 0.0411, 0.2338, 0.0661, 0.0270, 0.1493, 0.3857, 0.01576, 0.514])).cuda() 
                    weights = torch.Tensor([0.0483, 0.1208, 0.4725, 0.0411, 0.2338, 0.0661, 0.0270, 0.1493, 0.3857, 0.01576, 0.514]).cuda() 
                    weights_tensor = torch.where(torch.eq(var_class_target, 1), (var_class_target-weights), weights)
                    lossvalue = (weights_tensor * self.class_loss(varOutput, var_class_target) * c_weights).mean()
                
                else: 
                    lossvalue = self.class_loss(varOutput, var_class_target).mean()
                
                lossVal += lossvalue
                lossValNorm += 1
                
        outLoss = lossVal / lossValNorm
        return outLoss
    
    #-------------------------------------------------------------------------------- 
    
    def test(self, dataLoader):   
        
        cudnn.benchmark = True

        if self.use_gpu:
            outGT = torch.FloatTensor().cuda()
            outPRED = torch.FloatTensor().cuda()

        else:
            outGT = torch.FloatTensor()
            outPRED = torch.FloatTensor()
       
        self.model.eval()
        
        with torch.no_grad():
            for i, input in enumerate(dataLoader):

                class_target = input[1].cuda()
                outGT = torch.cat((outGT, class_target), 0).cuda()

                bs, c, h, w = input[0].size()
                varInput = input[0].view(-1, c, h, w).cuda(non_blocking = True)
            
                out = self.model(varInput)
                outPRED = torch.cat((outPRED, out), 0)
        
        return outGT, outPRED,

    #-------------------------------------------------------------------------------- 
    
    def inference(self, dataLoader):   
        
        cudnn.benchmark = True

        if self.use_gpu:
            outPRED = torch.FloatTensor().cuda()
        else:
            outPRED = torch.FloatTensor()
       
        self.model.eval()
        
        with torch.no_grad():
            for i, (input) in enumerate(dataLoader):

                bs, c, h, w = input[0].size()
                varInput = input[0].view(-1, c, h, w).cuda(non_blocking = True)
            
                out = self.model(varInput)
                outPRED = torch.cat((outPRED, out), 0)

        return outPRED
    
    #-------------------------------------------------------------------------------- 
    
    def extract_features(self, dataLoader, full = False):   
        
        cudnn.benchmark = True

        if self.use_gpu:
            outPRED = torch.FloatTensor().cuda()
        else:
            outPRED = torch.FloatTensor()
       
        self.model.eval()
        
        with torch.no_grad():
            for i, input in enumerate(dataLoader):

                bs, c, h, w = input[0].size()
                varInput = input[0].view(-1, c, h, w).cuda(non_blocking = True)
            
                features = self.model.module.densenet121.features(varInput.cuda())
                features = nn.functional.relu(features)
                if full:
                    features = torch.flatten(features, 1) 
                    outPRED = torch.cat((outPRED, features), 0)
                else:
                    features = nn.functional.adaptive_avg_pool2d(features, (3, 3))
                    features = torch.flatten(features, 1) 
                    outPRED = torch.cat((outPRED, features), 0)

        return outPRED
    

class HeatmapGenerator ():
    
    #---- pathModel - path to the trained densenet model
    #---- class_names - Label names
    #---- size - input image size
    #---- transformSequence - Expected preprocessing pipeline
 
    def __init__ (self, pathModel, class_names, size, transformSequence, weights = None, use_gpu= True):
       
        #---- Initialize the network
        model = DenseNet121(10).cuda()
        
        if use_gpu:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = torch.nn.DataParallel(model)
        
        modelCheckpoint = torch.load(pathModel)
        model.load_state_dict(modelCheckpoint['state_dict'])
        
        self.class_names = class_names
        self.model = model
        self.model.eval()
        self.size = size
        self.use_gpu = use_gpu
        
        #---- Initialize the weights
        if weights is None:
            self.weights = list(self.model.module.densenet121.classifier.parameters())[-2]
            print(self.weights.shape)
        else :
            self.weights = weights
            print(self.weights.shape)
        
        self.transformSequence = transformSequence
    
    #--------------------------------------------------------------------------------
     
    def generate (self, pathImageFile, pathOutputFile, label):
        
        #---- Load image, transform, convert 
        with torch.no_grad():
 
            imageData = Image.open(pathImageFile).convert('RGB')
            imageData = self.transformSequence(imageData)
            imageData = imageData.unsqueeze_(0)
            if self.use_gpu:
                imageData = imageData.cuda()
            l = self.model(imageData)
            output = nn.functional.relu(self.model.module.densenet121.features(imageData))
            print(l)

            index = self.class_names.index(label)
            label_weights = self.weights[index]
            #---- Generate heatmap
            heatmap = None
            for i in range (0, len(label_weights)):
                map = output[0,i,:,:]
                if i == 0: heatmap = label_weights[i] * map
                else: heatmap += label_weights[i] * map
                npHeatmap = heatmap.cpu().data.numpy()

        #---- Blend original and heatmap 
                
        imgOriginal = cv2.imread(pathImageFile, 1)
        imgOriginal = cv2.resize(imgOriginal, (self.size, self.size))
        
        cam = npHeatmap - np.min(npHeatmap)
        cam /= np.max(cam)
#        cam *= l[index]
        cam = cv2.resize(cam, (self.size, self.size))
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        
        img = cv2.addWeighted(imgOriginal,1,heatmap,0.2,0)            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.title(label)
        plt.imshow(img)
        plt.plot()
        plt.axis('off')
        plt.savefig(pathOutputFile)
        plt.show()