import h5py
import zipfile
import imageio
import os
import numpy
import matplotlib.pyplot as plt
import pandas
import torch
from torch import nn
from torch.utils.data import Dataset

from Helper_Functions import View, generate_random_image
from model.CelebADataset import CelebADataset

# 检查可用GPU
if torch.cuda.is_available():
  torch.set_default_tensor_type(torch.cuda.FloatTensor)
  print("using cuda:", torch.cuda.get_device_name(0))
  pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建数据集对象
celeba_dataset = CelebADataset('img\celeba_dataset\celeba_aligned_small.h5py')
# 检查图像中的数据
# celeba_dataset.plot_image(43)
# 构建鉴别器
class Discriminator(nn.Module):

  def __init__(self):
    # initialise parent pytorch class
    super().__init__()

    # define neural network layers
    self.model = nn.Sequential(
      View(218 * 178 * 3),  # 作用是将尺寸为 (218, 178, 3) 的三维图像张量转换为尺寸为 (218 * 178*3) 的一维张量；

      nn.Linear(3 * 218 * 178, 100),
      nn.LeakyReLU(),

      nn.LayerNorm(100),

      nn.Linear(100, 1),
      nn.Sigmoid()
    )

    # create loss function
    self.loss_function = nn.BCELoss()

    # create optimiser, simple stochastic gradient descent
    self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)

    # counter and accumulator for progress
    self.counter = 0;
    self.progress = []

    pass

  def forward(self, inputs):
    # simply run model
    return self.model(inputs)

  def train(self, inputs, targets):
    # calculate the output of the network
    outputs = self.forward(inputs)

    # calculate loss
    loss = self.loss_function(outputs, targets)

    # increase counter and accumulate error every 10
    self.counter += 1;
    if (self.counter % 10 == 0):
      self.progress.append(loss.item())
      pass
    if (self.counter % 1000 == 0):
      print("counter = ", self.counter)
      pass

    # zero gradients, perform a backward pass, update weights
    self.optimiser.zero_grad()
    loss.backward()
    self.optimiser.step()

    pass

  def plot_progress(self):
    df = pandas.DataFrame(self.progress, columns=['loss'])
    df.plot(ylim=(0), figsize=(16, 8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5, 1.0, 5.0))
    pass

  pass

# 测试鉴别器
D = Discriminator()
# move model to cuda device
D.to(device)

for image_data_tensor in celeba_dataset:
    # real data
    D.train(image_data_tensor, torch.cuda.FloatTensor([1.0]))
    # fake data
    D.train(generate_random_image((218,178,3)), torch.cuda.FloatTensor([0.0]))
    pass