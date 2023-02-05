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
要实现两个functions
__len__返回数据集整体样本数量
__getitem__返回一个指定的样本

2. Data preprocessing:
Image resize 224 for ResNet18
Data Argumentation: rotate, crop
Normalize: mean, std
ToTensor

五个子文件夹，用各自label命名
"""

class Pokemon(Dataset):  # 这个类根据index来加载单个图片，可以用DataLoader来加载一个batch的图片
    def  __init__(self, root, resize, mode):
        super(Pokemon, self).__init__()
        self.root = root
        self.resize = resize

        self.name2label = {}  # "xxx" : 0, "yyy" : 1
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):  # 文件和目录都包涵，这里过滤掉文件
                continue
            self.name2label[name] = len(self.name2label.keys())

        # print(self.name2label)

        self.images, self.labels = self.load_csv("images.csv")

        # 对数据集进行裁剪：train, valid, test取不同成分
        if mode == 'train':
            self.images = self.images[: int(0.6*len(self.images))]
            self.labels = self.labels[: int(0.6 * len(self.labels))]
        elif mode == 'val':
            self.images = self.images[int(0.6 * len(self.images)): int(0.8 * len(self.images))]
            self.labels = self.labels[int(0.6 * len(self.labels)): int(0.8 * len(self.labels))]
        else:
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]


    def load_csv(self, filename):  # 得到数据对: image_path, image_label (不存image本身)

        if not os.path.exists(os.path.join(self.root, filename)):  # 如果csv文件不存在才需要写进来
            images = []
            for name in self.name2label.keys():
                # 把所有图片合并到一个list里，用path判断类别信息
                # 'pokemon/mewtwo/000001.png'
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
            # 1165张图片, 存到一个csv文件里
            print(len(images), images)

            random.shuffle(images) # 把所有图片随机打乱
            with open(os.path.join(self.root, filename), mode="w", newline="") as f:
                writer = csv.writer(f)
                for img in images:   # 'pokemon/mewtwo/000001.png'
                    name = img.split(os.sep)[-2]  # 分隔开之后取倒数第二个是name
                    label = self.name2label[name]
                    # 存储方式：'pokemon/mewtwo/000001.png', 2
                    writer.writerow([img, label])
                print('writen into csv file:', filename)

        # read from csv file（有csv文件就直接读）
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
        img, label = self.images[idx], self.labels[idx]  # 得到的是图片路径不是图片

        tf = transforms.Compose([
            lambda x: Image.open(x).convert("RGB"),   # change image path(string) to image data
            transforms.Resize((int(self.resize*1.25), int(self.resize*1.25))),  # 为了data augmentation稍微大一点
            # Data Argumentation
            transforms.RandomRotation(30),  # rotate太大可能造成网络不收敛
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],   # 统计的imageNet上图片的数据rgb均值方差
                                 std=[0.229, 0.224, 0.225])    # 把0-1之间变成-1-1之间
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

    # 可视化一个sample
    import visdom
    import time

    vis = visdom.Visdom()

    # db = Pokemon('pokeman', 224, "train")
    #
    # # 加载一张图片
    # x, y = next(iter(db))
    # print('sample:', x.shape, y.shape, y)
    # vis.image(db.denormalize(x), win='sample-x', opts=dict(title='sample-x'))
    #
    # # 加载一个batch的图片
    # loader = DataLoader(db, batch_size=32, shuffle=True)
    # for x, y in loader:
    #     vis.images(db.denormalize(x), nrow=8, win='batch', opts=dict(title='batch-x'))
    #     vis.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))
    #
    #     time.sleep(10)

    # 如果用标准方式存储图片Dataset(一个文件夹下图片都是一类)
    # 通用方法就是上面实现的class，要先拿到每个图片的path，label对存到csv文件里，然后从里面去sample image和对应label(getitem)
    # 最后放到dataloader里操作
    tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    db = torchvision.datasets.ImageFolder(root='pokeman', transform=tf)   # 直接用API：传入路径名和变化器就行
    loader = DataLoader(db, batch_size=32, shuffle=True, num_workers=8)   # 可以用多线程加速

    print(db.class_to_idx)  # 直接得到map

    for x, y in loader:
        vis.images(x, nrow=8, win='batch', opts=dict(title='batch-x'))
        vis.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))  # y从tensor到numpy再到string

        time.sleep(10)

if __name__ == '__main__':
    main()