"""
为了处理 CelebA 数据集中的数据，需要对之前使用的 MNIST 图像的数据集类进行简单修改
"""
import h5py
import numpy
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from torch.utils.data import Dataset

class CelebADataset(Dataset):

    def __init__(self, file):
        self.file_object = h5py.File(file, 'r')
        self.dataset = self.file_object['img_align_celeba']
        pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if (index >= len(self.dataset)):
            raise IndexError()
        img = numpy.array(self.dataset[str(index) + '.jpg'])
        return torch.FloatTensor(img) / 255.0

    def plot_img(self, index):
        plt.imshow(numpy.array(self.dataset[str(index) + '.jpg']), interpolation='nearest')
        pass

    pass


# 创建数据集对象
celeba_dataset = CelebADataset('..\img\celeba_dataset\celeba_aligned_small.h5py')
