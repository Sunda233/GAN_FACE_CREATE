'''
对图片进行预处理

'''
import torch
import torch.nn as nn
from imageio import imread
from torch.utils.data import Dataset
import h5py
import pandas, numpy, random
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条可视化库
from PIL import Image
import numpy as np

# check if CUDA is available
# if yes, set default tensor type to cuda

# 如果GPU可用，则选择GPU
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print("using cuda:", torch.cuda.get_device_name(0))
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# device

# 辅助函数，可以从numpy图像中裁剪任意大小的图像，裁剪区域位于图像的正中央。
def crop_centre(img,new_width,new_height):
    height,width, _ = img.shape
    startx = width // 2 - new_width//2
    starty = height // 2 - new_height//2
    return img[starty:starty+new_height,startx:startx+new_width, :]

# dataset class
# 对数据集预处理
class CelebADataset(Dataset):

    def __init__(self, file):
        self.file_object = h5py.File(file, 'r')
        self.dataset = self.file_object['img_align_celeba']
        pass

    def __len__(self):
        return len(self.dataset)
    # __getitem__ 返回一个张量（通道大小，通道，高度，宽度）
    # numpy 三维张量（高度，宽度，3）
    # permute（2,0,1）numpy数组重新排序，（3，高度，宽度）
    def __getitem__(self, index):
        if (index >= len(self.dataset)):
            raise IndexError()
        img = numpy.array(self.dataset[str(index) + '.jpg'])
        # 裁剪图像，变为128*128规格的图像
        img = crop_centre(img,128,128)
        return torch.cuda.FloatTensor(img).permute(2, 0, 1).view(1, 3, 128, 128) / 255.0
        # view 为批次大小设置一个额外的维度，为1
    def plot_image(self, index):
        img = numpy.array(self.dataset[str(index)+'.jpg'])
        img = crop_centre(img, 128, 128)  # 裁剪图形
        plt.imshow(img, interpolation='nearest')
        # plt.show()
        pass

    pass


celeba_dataset = CelebADataset('img\celeba_dataset\celeba_aligned_small.h5py')

celeba_dataset.plot_image(49)
print(celeba_dataset.plot_image(49))
plt.show()

'''
预处理——2
'''
L_image=Image.open(celeba_dataset.plot_image(49))
out = L_image.convert("RGB")
img=np.array(out)

print(out.size)
print(img.shape)#高 宽 三原色分为三个二维矩阵
print(img)
