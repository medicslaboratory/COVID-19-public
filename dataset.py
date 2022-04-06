# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 12:50:46 2020

@author: gosek
"""
import sys
import os
import torch
import csv
import cv2
import pymysql.cursors
from torch.utils.data import Dataset
from PIL import Image


import xml.dom.minidom

class CheXpertDataSet(Dataset):
    def __init__(self, image_list_file, transform=None, policy="ones"):
        """
        image_list_file: path to the file containing images with corresponding labels.
        transform: optional transform to be applied on a sample.
        policy: name the policy with regard to the uncertain labels
        """
        image_names = []
        labels = []
        ages = []
        sexes= []

        with open(image_list_file, "r") as f:
            csvReader = csv.reader(f)
            next(csvReader, None)
            
            for line in csvReader:
                image_name= line[0]
                label = [line[i] for i in [6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 18]]
                sex = [1 if line[1] == 'Male' else 0]
                age = [float(line[2])/100]
                
                for i in range(11):
                    if label[i]:
                        a = float(label[i])
                        if a == 1:
                            label[i] = 1
                        elif a == -1:
                            if policy == "ones":
                                label[i] = 1
                                
                            elif policy == "zeroes":
                                label[i] = 0
                            else:
                                label[i] = 0
                        else:
                            label[i] = 0
                    else:
                        label[i] = 0
                        
                image_names.append(os.path.join('/mnt/data', image_name))
                labels.append(label)
                ages.append(age)
                sexes.append(sex)

        self.image_names = image_names
        self.labels = labels
        self.ages = ages
        self.sexes = sexes
        self.transform = transform

    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        
        image_name = self.image_names[index]
        im = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        im = cv2.equalizeHist(im)
        image = Image.fromarray(im).convert('RGB')
        class_label = self.labels[index]
        sex_label = self.sexes[index]
        age_label = self.ages[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(class_label), torch.FloatTensor(age_label), torch.FloatTensor(sex_label)

    def __len__(self):
        return len(self.image_names)


class CovidDataSet(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        image_dir: path to the directory containing images (no labels atm)
        transform: optional transform to be applied on a sample.
        Upolicy: name the policy with regard to the uncertain labels
        """
        self.image_dir = image_dir
        self.image_names = os.listdir(image_dir)
        self.transform = transform

    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        
        image_name = self.image_names[index]
        im = cv2.imread('/mnt/Open_COVID/Images/'+ image_name, cv2.IMREAD_GRAYSCALE)
        im = cv2.equalizeHist(im)
        image = Image.fromarray(im).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, 0

    def __len__(self):
        return len(self.image_names)

class DBDataSet(Dataset):
    def __init__(self, image_list, labels_list, age_list, sex_list, transform=None):
        """
        image_dir: path to the directory containing images (no labels atm)
        transform: optional transform to be applied on a sample.
        Upolicy: name the policy with regard to the uncertain labels
        """
        self.image_names = image_list
        self.image_labels = labels_list
        self.age = age_list
        self.sex = sex_list
        self.transform = transform

    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        
        image_name = self.image_names[index]
        label= self.image_labels[index]
        age= self.age[index]
        sex= self.sex[index]
        if os.path.isfile('/mnt' + image_name[:-4] + '_crop.jpg'):
            im = cv2.imread('/mnt' + image_name[:-4] + '_crop.jpg', cv2.IMREAD_GRAYSCALE)
        else:
            im = cv2.imread('/mnt' + image_name, cv2.IMREAD_GRAYSCALE)

        print(image_name)
        image = cv2.equalizeHist(im)
        image = Image.fromarray(im).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label), torch.FloatTensor(age), torch.FloatTensor(sex)

    def __len__(self):
        return len(self.image_names)

# Connect to the database
def execute_query(sql):
	# Get xml info
    lexml = xml.dom.minidom.parse('config.xml')
    host = lexml.getElementsByTagName('host')[0].firstChild.data
    db = lexml.getElementsByTagName('db')[0].firstChild.data
    user = lexml.getElementsByTagName('user')[0].firstChild.data
    passwd = lexml.getElementsByTagName('pass')[0].firstChild.data

    connection = pymysql.connect(host=host,
                                 user=user,
                                 password=passwd,
                                 db=db,
                                 charset='utf8',
                                 cursorclass=pymysql.cursors.DictCursor)
    try:
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
            # Read a single record
#	        sql = "SELECT * FROM imaging"
            cursor.execute(sql)
            result = cursor.fetchall()
    except pymysql.Error:
            raise RuntimeError(
                "Cannot connect to database. ")
    finally:
        connection.close()
        return(result)