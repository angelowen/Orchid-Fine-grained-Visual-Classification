from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import pandas as pd
import os
from os import listdir

class TrainOrchidDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.x = []
        self.y = []
        self.transform = transform
        sample_submission = pd.read_csv(os.path.join(root_dir,'train_label.csv'))
        for idx, filename in enumerate(sample_submission['filename']):
            # print("filename: ", filename,"Label: ",sample_submission['category'][idx])
            filename = os.path.join(root_dir,filename)
            self.x.append(filename)
            self.y.append(sample_submission['category'][idx])
        self.num_classes = max(self.y)+1
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image = Image.open(self.x[index]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.y[index]
    
class ValOrchidDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.x = []
        self.y = []
        self.transform = transform
        sample_submission = pd.read_csv(os.path.join(root_dir,'val_label.csv'))
        for idx, filename in enumerate(sample_submission['filename']):
            filename = os.path.join(root_dir,filename)
            self.x.append(filename)
            self.y.append(sample_submission['category'][idx])
        self.num_classes = max(self.y)+1
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image = Image.open(self.x[index]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.y[index]
    
class TestOrchidDataset(Dataset): # if no submission file (.csv)
    def __init__(self, root_dir, transform=None):
        self.x = []
        self.y = []
        self.transform = transform
        files = listdir(root_dir)
        for f in files:
            filename = os.path.join(root_dir, f)
            self.x.append(filename)
            self.y.append(0)
              
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image = Image.open(self.x[index]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.y[index]
    
class CompOrchidDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.x = []
        self.y = []
        self.transform = transform
        sample_submission = pd.read_csv('Orchid_Final_ds/submission_template.csv')
        for idx, filename in enumerate(sample_submission['filename']):
            filename = os.path.join(root_dir,filename)
            self.x.append(filename)
            self.y.append(sample_submission['category'][idx])
        self.num_classes = max(self.y)+1
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image = Image.open(self.x[index]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.y[index]