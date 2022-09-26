"""
CelebA数据集图像：217*178
为了CNN简化为128*128
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import h5py
import pandas, numpy, random
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条可视化库

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

celeba_dataset.plot_image(43)





# functions to generate random data

def generate_random_seed(size):
    random_data = torch.randn(size)
    return random_data


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape,

    def forward(self, x):
        return x.view(*self.shape)

# 推送测试

# discriminator class
# 定义判别器
class Discriminator(nn.Module):

    def __init__(self):
        # initialise parent pytorch class
        super().__init__()

        # define neural network layers
        self.model = nn.Sequential(
            # 预期的输入形状（1,3,128,128）
            nn.Conv2d(3, 256, kernel_size=8, stride=2),
            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.Conv2d(256, 700, kernel_size=8, stride=2),
            nn.BatchNorm2d(700),
            nn.GELU(),

            nn.Conv2d(700, 3, kernel_size=8, stride=2),
            nn.GELU(),

            # 用view将特征图重塑为一个简单地一维张量。
            View(3*10*10),
            nn.Linear(3*10*10, 1),
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
        # if (self.counter % 1000 == 0):
        #     print("counter = ", self.counter)
        #     pass

        # zero gradients, perform a backward pass, update weights
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        pbar.update(1)
        pass

    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['D_loss'])
        df.plot(ylim=(0), figsize=(16, 8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5, 1.0, 5.0))
        pass

    pass

# generator class
#
# 定义生成器
class Generator(nn.Module):
    def __init__(self):
        # initialise parent pytorch class
        super().__init__()

        # define neural network layers
        self.model = nn.Sequential(
            # 输入是一个一维数组
            # 100个随机种子-》3*1*11张量-》1,3,11,11张量
            nn.Linear(100, 3 * 11 * 11),
            nn.GELU(),
            # nn.LeakyReLU(0.2),

            # 转换成四维，利用View
            View((1, 3, 11, 11)),

            # 通过转置卷积（反卷积）层
            nn.ConvTranspose2d(3, 256, kernel_size=8, stride=2),
            nn.BatchNorm2d(256),
            nn.GELU(),
            # nn.LeakyReLU(0.2),

            # 第二层反卷积
            nn.ConvTranspose2d(256, 700, kernel_size=8,stride=2),
            nn.BatchNorm2d(700),
            nn.GELU(),
            # nn.LeakyReLU(0.2),

            # 最后一层转置卷积层，此时需要额外设置，补全padding = 1 ，作用：从中间网格中去掉外围的网格。若没有补全，想要正确输出则需要增加额外参数。
            nn.ConvTranspose2d(700, 3, kernel_size=8, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )

        # create optimiser, simple stochastic gradient descent
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)
        # counter and accumulator for progress
        self.counter = 0
        self.progress = []

        pass

    # 上述代码简单获得输入并传递给使用nn.Sequential() 定义的 self.model() 模型中。模型的输出可以返回给任何调用 forward() 函数的地方
    def forward(self, inputs):
        # simply run model
        return self.model(inputs)

    def train(self, D, inputs, targets):
        g_output = self.forward(inputs)  # 计算网络输出
        d_output = D.forward(g_output)  # 传递到鉴别器
        loss = D.loss_function(d_output, targets)  # 计算差距
        # 增加计数器并每10次累积错误，为了可视化保存值
        self.counter += 1
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass
        self.optimiser.zero_grad()  # 梯度归零
        loss.backward()  # 反向传递
        self.optimiser.step()  # 更新权重


        pass

    # 绘制图像
    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['G_loss'])
        df.plot(ylim=(0), figsize=(16, 8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5, 1.0, 5.0))
        pass

    pass

# train


# 进行训练
D = Discriminator()
D.to(device)
G = Generator()
G.to(device)

epochs = 5
with tqdm(total=epochs * celeba_dataset.__len__()*2) as pbar:
    for epoch in range(epochs):
        # print("epoch = ", epoch + 1)
    # 训练生成器和判别器
        for image_data_tensor in celeba_dataset:
            # train discriminator on true
            D.train(image_data_tensor, torch.cuda.FloatTensor([1.0]))
            # train discriminator on false
            # use detach() so gradients in G are not calculated
            D.train(G.forward(generate_random_seed(100)).detach(), torch.cuda.FloatTensor([0.0]))

            # train generator
            G.train(D, generate_random_seed(100), torch.cuda.FloatTensor([1.0]))
        pass
    pass

# 绘图
# plot discriminator error
D.plot_progress()
plt.show()
# plot generator error
G.plot_progress()
plt.show()

# plot several outputs from the trained generator
# plot a 3 column, 2 row array of generated images
f, axarr = plt.subplots(2, 3, figsize=(16, 8))
for i in range(2):
    for j in range(3):
        output = G.forward(generate_random_seed(100))
        img = output.detach().permute(0, 2, 3, 1).view(128, 128, 3).cpu().numpy()
        axarr[i, j].imshow(img, interpolation='none', cmap='Blues')
        pass
    pass
plt.show()

# 当前分配给张量的内存大小
# torch.cuda.memory_allocated(device) / (1024*1024*1024)
# 当前分配给张量的内存总量
# torch.cuda.max_memory_allocated(device) / (1024*1024*1024)
# 内存消耗汇总
# print(torch.cuda.memory_summary(device, abbreviated=True))
