from __future__ import print_function
import torch
from torch.utils.data import Dataset
import glob
import os
import numpy as np
import csv
from collections import Counter
from torchvision import transforms
from PIL import Image
import pandas as pd
import random


def get_transform(name):

    if 'train' in name:
        data_transforms = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=0.5),
             transforms.Resize([256, 256]),
             transforms.RandomCrop([224, 224]),
             transforms.ToTensor(),
             transforms.Normalize([0.485], [0.229])
             ])
    else:
        data_transforms = transforms.Compose(
            [transforms.Resize([256, 256]),
             transforms.CenterCrop([224, 224]),
             transforms.ToTensor(),
             transforms.Normalize([0.485], [0.229])
             ])
    return data_transforms

class Boneage_Dataset(Dataset):
    def __init__(self, name, args, data_batch,switch):
        super(Boneage_Dataset, self).__init__()
        assert name in ['train', 'val','test','test_final_loader']
        self.name = name
        self.args = args
        self.transform = get_transform(name)

        # Loading labels
        df = pd.read_csv('data/total_labels.csv')
        df['id'] = df['id'].apply(str)
        self.labels = dict(zip(df.id, df.boneage))

                
    
        if name == 'test_final_loader':
            data = list(pd.read_csv(args.data_dir + "/total_test.csv", header=None)[0])
        else:
            data = list(pd.read_csv(args.data_dir + "/total_train.csv", header=None)[0])


        data = [str(image) for image in data]
        random.Random(args.seed).shuffle(data)



        j = data_batch
        
        train_size = int(args.train_size*0.8/(args.sites))
        val_size = int(train_size/4)
        if name == 'train':
            data = data[(train_size+val_size)*(j-1):(train_size+val_size)*(j-1)+train_size] 
        if name == 'test':
            data = data[int(args.train_size/2)*j:int(args.train_size/2)*(j+1)] 
        if name == 'val': 
            data = data[(train_size+val_size)*(j-1)+train_size:(train_size+val_size)*(j-1)+train_size+val_size] 
        



        random.shuffle(data)

        # Loading images
        files = glob.glob(os.path.join('old_data/boneage-training-dataset/boneage-training-dataset', "*"))
        self.images = {}

        for file in files:
            filename = os.path.basename(os.path.splitext(file)[0])

            if filename in data: 
                self.images[filename] = Image.open(file)
        
        labels = {k: v for k,v in self.labels.items() if k in data}


        print("Label balance for " + name, Counter(labels.values()))
        self.set = list(self.images.keys())
        print(self.set)

    def __getitem__(self, idx):
        key = self.set[idx]
        return {'image': self.transform(self.images[key]),
                'label':  np.array([self.labels[key]]),
                'img_name': key}

    def __len__(self):
        return len(self.set)
