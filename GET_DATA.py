"""读取压缩包中的数据"""
import h5py
import zipfile
import imageio
import os
import numpy
import matplotlib.pyplot as plt

# 定位 HDF5 包的位置，可以放到其他位置，下同
hdf5_file = 'img\celeba_dataset\celeba_aligned_small.h5py'

# 从 202,599 个图像中提取 20,000 个图像并打包到 HDF5 中
total_images = 20000

with h5py.File(hdf5_file, 'w') as hf:

    count = 0

    with zipfile.ZipFile('img\celeba\img_align_celeba.zip', 'r') as zf:
      for i in zf.namelist():
        if (i[-4:] == '.jpg'):
          # 提取图像
          ofile = zf.extract(i)
          img = imageio.imread(ofile)
          os.remove(ofile)

          # 使用新的名字来将图像加入到 HDF5 中
          hf.create_dataset('img_align_celeba/'+str(count)+'.jpg', data=img, compression="gzip", compression_opts=9)

          count = count + 1
          if (count%1000 == 0):
            print("images done .. ", count)
            pass

          # stop when total_images reached
          if (count == total_images):
            break
          pass

        pass
      pass

# 观察数据,使用一个文件对象，在只读模式下打开 HDF5 文件，就像在 Python 中使用 open() 一样，通过循环文件对象，打印出文件顶端的组的名称：
with h5py.File('img\celeba_dataset\celeba_aligned_small.h5py', 'r') as file_object:
  for group in file_object:
    print(group)
    pass

# 查看其中的一张图像
with h5py.File('img\celeba_dataset\celeba_aligned_small.h5py', 'r') as file_object:
  dataset = file_object['img_align_celeba']
  image = numpy.array(dataset['7.jpg'])
  plt.imshow(image, interpolation='none')
  pass