# coding: utf-8
# Code is based on the work of https://github.com/gaetandi/cheXpert by Gaëtan Dissez & Guillaume Duboc

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
import sklearn.metrics as metrics

from dataset import CheXpertDataSet, CovidDataSet
from model_utils import Model, HeatmapGenerator

use_gpu = torch.cuda.is_available()

def visualize_loss(losst, losse):
    losstn = []
        
    for i in range(0, len(losst), 140):
        losstn.append(np.mean(losst[i:i+140]))
    
    print(losstn)
    print(losse)
    #
    
    lt = losstn
    batch = [i*140 for i in range(len(lt))]
    
    plt.plot(batch, lt, label = "train")
    plt.plot(batch, losse, label = "eval")
    plt.xlabel("Nb of batches (size_batch = 64)")
    plt.ylabel("BCE loss")
    plt.title("BCE loss evolution")
    plt.legend()
    
    plt.savefig("chart5.png", dpi=1000)
    plt.show()
    
def show_AUC(y_true, y_pred, class_names, out_path = ''):
    letters = ['a): ', 'b): ', 'c): ', 'd): ', 'e): ', 'h): ', 'g): ', 'f): ']
    ctr = 0
    for i in range(len(class_names)):
        plt.rcParams.update({'font.size': 6})
        if i not in [3,9]:
            ctr += 1
            fpr, tpr, threshold = metrics.roc_curve(y_true.cpu()[:,i], y_pred.cpu()[:,i])
            roc_auc = metrics.auc(fpr, tpr)
            if ctr not in [6,8]:
                f = plt.subplot(2, 4, ctr)
            elif ctr == 6:
                f = plt.subplot(2, 4, 8)
            elif ctr == 8:
                f = plt.subplot(2, 4, 6)
                
            plt.title(letters[ctr-1] + class_names[i])
            plt.plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)
            
            plt.legend(loc = 'lower right')
            plt.plot([0, 1], [0, 1],'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('Sensitivity')
            if ctr > 4:
                plt.xlabel('1 - Specificity')
#            plt.tight_layout() 
            plt.subplots_adjust( hspace  = 0.3, wspace = 0.5 )
    if len(out_path) > 0:
        plt.savefig(out_path, dpi=1000)
    plt.show()

if __name__ == "__main__":
    # Training settings: batch size, maximum number of epochs
    verbose = 3
    trBatchSize = 16
    trMaxEpoch = 3
    #If training the network
    train = False
    #If writing image features to csv
    write = False
    
    #path to the file containing ChexPert image
    if train:    
        pathFileTrain = 'train.csv'
        pathFileValid = 'valid.csv'

    #Path to the directory containing COVID-19 images (no need for labels)
    pathCovid= '/mnt/data/COVID'
    
    # Parameters related to image transforms: size of the down-scaled image, cropped image
    imgSize = 320
    
    # Class names
    class_names = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
                   'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                   'Pleural Effusion', 'Pleural Other']
    
    nnClassCount = len(class_names)
    
    #Imagenet normalization parameters
    #New normalization parameters
    normalize = transforms.Normalize([0.503, 0.503, 0.503], [0.2907, 0.2907, 0.2907])
    transformList_train = []
    transformList_train.append(transforms.Resize((imgSize,imgSize)))
    transformList_train.append(transforms.RandomCrop((imgSize-30,imgSize-30)))
    transformList_train.append(transforms.Resize((imgSize,imgSize)))
    transformList_train.append(transforms.ToTensor())
    transformList_train.append(normalize)
    transformSequence_train=transforms.Compose(transformList_train)
    
    transformList_test = []
    transformList_test.append(transforms.Resize((imgSize,imgSize)))
    transformList_test.append(transforms.ToTensor())
    transformList_test.append(normalize)
    transformSequence_test=transforms.Compose(transformList_test)
    
    #LOAD DATASETS
    covid_set = CovidDataSet(pathCovid, transformSequence_train)
    #policy = zeroes means that uncertain labels are set to 0
    datasetTrain = CheXpertDataSet(pathFileTrain ,transformSequence_train, policy="zeroes")
    datasetValid = CheXpertDataSet(pathFileValid, transformSequence_test)
    
    dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=0, pin_memory=True)
    dataLoaderVal = DataLoader(dataset=datasetValid, batch_size=trBatchSize, shuffle=False, num_workers=0, pin_memory=True)
    dataLoaderCovid = DataLoader(dataset=covid_set, num_workers=0, pin_memory=True)
    
    #checkpoint = None when initializing a new training
    inst_model = Model(n_classes = 10, checkpoint = None, use_gpu = True)
    
    if train:
        batch, losst, losse = inst_model.train(trMaxEpoch, dataLoaderTrain, dataLoaderVal, weighting = True)
        print("Model trained")
    
        if verbose > 0:
            visualize_loss(losst, losse)
    
    y_true, y_pred = inst_model.test(dataLoaderVal, class_names)
    rad_signs = inst_model.inference(dataLoaderCovid)
    
    if write:
        y_pred_cov = inst_model.extract_features(dataLoaderCovid, full = False).cpu().data.numpy()
         
        with open('results/covid_features.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for pred in y_pred_cov:
                writer.writerow(pred)
        
    with open('results/covid_signs.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for pred in rad_signs.cpu().numpy():
            writer.writerow(pred)

    if verbose > 1: 
        show_AUC(y_true, y_pred, class_names, 'results/ROC.png')
        
        pathInputImage = 'mnt/data/COVID/10093_1.jpg'
        pathOutputImage = 'results/heatmap.png'
        pathModel = 'm-epoch1-31082020-132716.pth.tar'
        
        h = HeatmapGenerator(pathModel, class_names, 320, transformSequence_test)
        
        h.generate(pathInputImage, pathOutputImage, label = 'Pneumonia')