"""查看压缩包中的数据"""
import h5py
import zipfile
import imageio
import os
import numpy
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

# 观察数据,使用一个文件对象，在只读模式下打开 HDF5 文件，就像在 Python 中使用 open() 一样，通过循环文件对象，打印出文件顶端的组的名称：
with h5py.File('img\celeba_dataset\celeba_aligned_small.h5py', 'r') as file_object:
  for group in file_object:
    print(group)
    pass

# 查看其中的一张图像
with h5py.File('img\celeba_dataset\celeba_aligned_small.h5py', 'r') as file_object:
  dataset = file_object['img_align_celeba']
  image = numpy.array(dataset['8.jpg'])
  plt.imshow(image, interpolation='none')
  plt.show()
  pass

# 为了处理 CelebA 数据集中的数据，需要对之前使用的 MNIST 图像的数据集类进行简单修改，
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

  def plot_image(self, index):
    plt.imshow(numpy.array(self.dataset[str(index) + '.jpg']), interpolation='nearest')
    pass

  pass

# 创建数据集对象
celeba_dataset = CelebADataset('img\celeba_dataset\celeba_aligned_small.h5py')

# 检查图像中的数据
celeba_dataset.plot_image(43) # 不可用



