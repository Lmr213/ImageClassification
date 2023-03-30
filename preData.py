import torch
import os
import glob
import random
import csv

import torchvision.datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

"""
1. Load data
inherit from torch.utils.data.Dataset
implement two functions
__len__
__getitem__

2. Data preprocessing:
Image resize 224 for ResNet18
Data Argumentation: rotate, crop
Normalize: mean, std
ToTensor

"""

class Pokemon(Dataset):  # resize:224 for ResNet
    def  __init__(self, root, resize, mode):
        super(Pokemon, self).__init__()
        self.root = root
        self.resize = resize

        self.name2label = {}  # "xxx" : 0, "yyy" : 1
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):  # for filter
                continue
            self.name2label[name] = len(self.name2label.keys())

        self.images, self.labels = self.load_csv("images.csv")

        # split dataset
        if mode == 'train':
            self.images = self.images[: int(0.6*len(self.images))]
            self.labels = self.labels[: int(0.6 * len(self.labels))]
        elif mode == 'val':
            self.images = self.images[int(0.6 * len(self.images)): int(0.8 * len(self.images))]
            self.labels = self.labels[int(0.6 * len(self.labels)): int(0.8 * len(self.labels))]
        else:
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]

    
    def load_csv(self, filename):  # store data pair into a csv file: image_path, image_label 

        if not os.path.exists(os.path.join(self.root, filename)):  # if csv file does not exist
            images = []
            for name in self.name2label.keys():
                # put all images path into a list and use path name to classify
                # 'pokemon/mewtwo/000001.png'
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
          
            random.shuffle(images) 
            with open(os.path.join(self.root, filename), mode="w", newline="") as f:
                writer = csv.writer(f)
                for img in images:   # 'pokemon/mewtwo/000001.png'
                    name = img.split(os.sep)[-2]  # second last is name
                    label = self.name2label[name]
                    # way to store：'pokemon/mewtwo/000001.png', 2
                    writer.writerow([img, label])

        # read from csv file
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                # 'pokemon/mewtwo/000001.png', 2
                img, label = row
                label = int(label)

                images.append(img)
                labels.append(label)

        assert len(images) == len(labels)
        return images, labels


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # idx range from 0 to len(images)
        # self.images, self.labels
        img, label = self.images[idx], self.labels[idx]  # get iamge path not image

        tf = transforms.Compose([
            lambda x: Image.open(x).convert("RGB"),   # change image path(string) to image data
            transforms.Resize((int(self.resize*1.25), int(self.resize*1.25))),  # for data augmentation
            # Data Argumentation
            transforms.RandomRotation(30), 
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # from imageNet
                                 std=[0.229, 0.224, 0.225])   # change from 0-1 to -1-1
        ])

        img = tf(img)
        label = torch.tensor(label)

        return img, label
    
    def denormalize(self, x_hot):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # x: [c, h, w]
        # mean: [3] => [3, 1, 1]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)  # 插入两个维度 1，1
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x = (x_hot * std) + mean
        return x


def main():
    
    import visdom
    import time

    vis = visdom.Visdom()

    db = Pokemon('pokeman', 224, "train")
    
    # load an image
    x, y = next(iter(db))
    print('sample:', x.shape, y.shape, y)
    vis.image(db.denormalize(x), win='sample-x', opts=dict(title='sample-x'))
    
    # load image batch
    loader = DataLoader(db, batch_size=32, shuffle=True)
    for x, y in loader:
        vis.images(db.denormalize(x), nrow=8, win='batch', opts=dict(title='batch-x'))
        vis.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))
    
        time.sleep(10)

    # 如果用标准方式存储图片Dataset(一个文件夹下图片都是一类)
    # 通用方法就是上面实现的class，要先拿到每个图片的path，label对存到csv文件里，然后从里面去sample image和对应label(getitem)
    # 最后放到dataloader里操作
    # tf = transforms.Compose([
    #     transforms.Resize((128, 128)),
    #     transforms.ToTensor()
    # ])
    # db = torchvision.datasets.ImageFolder(root='pokeman', transform=tf)   
    # loader = DataLoader(db, batch_size=32, shuffle=True, num_workers=8)  

    for x, y in loader:
        vis.images(x, nrow=8, win='batch', opts=dict(title='batch-x'))
        vis.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))  # y from tensor to numpy to string

        time.sleep(10)

if __name__ == '__main__':
    main()
