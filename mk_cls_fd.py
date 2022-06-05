import pandas as pd
import os
import shutil
import numpy as np
from PIL import Image
from pathlib import Path
def get_class_fig():
    class_label = 0
    df = pd.read_csv ('dataset/label.csv')
    for name,item in zip(df['filename'],df['category']):
        if item == class_label:
            print(name,item)
            shutil.copyfile(os.path.join('./dataset',name), os.path.join('./class0',name))
            
def all_class_fig():
    folder = "folder_class_ds"
    if not os.path.exists(folder):
        os.makedirs(folder)
    df = pd.read_csv ('dataset/label.csv')
    for name,label in zip(df['filename'],df['category']):
        cls_folder = os.path.join(folder,str(label))
        if not os.path.exists(cls_folder):
            os.makedirs(cls_folder)
        shutil.copyfile(os.path.join('./dataset',name), os.path.join(cls_folder,name))
        
def img2np(size):
    root_dir = 'folder_class_ds'
    total = []
    for i, _dir in enumerate(sorted(Path(root_dir).glob('*'))):
        class_img = []
        for file in _dir.glob('*.jpg'):
            img_file = Image.open(file)
            w,h = img_file.size
            if h==480:
                pos = (320-size/2,240-size/2,320+size/2,240+size/2)
                # pos = (256,176,384,304) # (left, top, right, bottom)
            else:
                pos = (240-size/2,320-size/2,240+size/2,320+size/2)
                # pos = (176,256,304,384) 
            img_file = img_file.crop(pos)
            class_img.append(np.array(img_file))
        print(_dir,np.array(class_img).shape)
        total.append(np.array(class_img))
    x = np.array(total)
    print(x.shape)
    np.save(f'orchid_dataset_{size}.npy', x)
            
if __name__ == "__main__":
    img2np(size = 256)
