
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """残差模块"""
    def __init__(self ,inChannals ,outChannals):
        """初始化残差模块"""
        super(ResBlock ,self).__init__()
        self.conv1 = nn.Conv2d(inChannals ,outChannals ,kernel_size=1 ,bias=False)
        self.bn1 = nn.BatchNorm2d(outChannals)  # # 数据归一化 保证每个特征被分类器平等对待 防止小数据被大数据吃掉
        self.conv2 = nn.Conv2d(outChannals ,outChannals ,kernel_size=3 ,stride=1 ,padding=1 ,bias=False)
        self.bn2 = nn.BatchNorm2d(outChannals)
        self.conv3 = nn.Conv2d(outChannals ,outChannals ,kernel_size=1 ,bias=False)
        self.relu = nn.PReLU()

    def forward(self ,x):
        """前向传播过程"""
        resudial = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += resudial
        out = self.relu(out)
        return out



class Generator(nn.Module):
    """生成模型(4x)"""
    def __init__(self):
        """初始化模型配置"""
        super(Generator ,self).__init__()
        # 卷积模块1
        self.conv1 = nn.Conv2d(3 ,64 ,kernel_size=9 ,padding=4 ,padding_mode='reflect' ,stride=1)
        self.relu = nn.PReLU()
        # 残差模块
        self.resBlock = self._makeLayer_(ResBlock ,64 ,64 ,5)
        # 卷积模块2
        self.conv2 = nn.Conv2d(64 ,64 ,kernel_size=1 ,stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.PReLU()

        # 子像素卷积
        self.convPos1 = nn.Conv2d(64 ,256 ,kernel_size=3 ,stride=1 ,padding=2 ,padding_mode='reflect')
        self.pixelShuffler1 = nn.PixelShuffle(2)
        self.reluPos1 = nn.PReLU()

        self.convPos2 = nn.Conv2d(64 ,256 ,kernel_size=3 ,stride=1 ,padding=1 ,padding_mode='reflect')
        self.pixelShuffler2 = nn.PixelShuffle(2)
        self.reluPos2 = nn.PReLU()

        self.finConv = nn.Conv2d(64 ,3 ,kernel_size=9 ,stride=1)

    def _makeLayer_(self ,block ,inChannals ,outChannals ,blocks):
        """构建残差层"""
        layers = []
        layers.append(block(inChannals ,outChannals))

        for i in range(1 ,blocks):
            layers.append(block(outChannals ,outChannals))

        return nn.Sequential(*layers)

    def forward(self ,x):
        """前向传播过程"""
        x = self.conv1(x)
        x = self.relu(x)
        residual = x
        out = self.resBlock(x)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.convPos1(out)
        out = self.pixelShuffler1(out)
        out = self.reluPos1(out)
        out = self.convPos2(out)
        out = self.pixelShuffler2(out)
        out = self.reluPos2(out)
        out = self.finConv(out)

        return out

    class ConvBlock(nn.Module):
        """残差模块"""

        def __init__(self, inChannals, outChannals, stride=1):
            """初始化残差模块"""
            super(ConvBlock, self).__init__()
            self.conv = nn.Conv2d(inChannals, outChannals, kernel_size=3, stride=stride, padding=1,
                                  padding_mode='reflect', bias=False)
            self.bn = nn.BatchNorm2d(outChannals)  # 数据归一化 保证每个特征被分类器平等对待 防止小数据被大数据吃掉
            self.relu = nn.LeakyReLU()

        def forward(self, x):
            """前向传播过程"""
            out = self.conv(x)
            out = self.bn(out)
            out = self.relu(out)
            return out

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
            self.relu1 = nn.LeakyReLU()

            self.convBlock1 = ConvBlock(64, 64, stride=2)
            self.convBlock2 = ConvBlock(64, 128, stride=1)
            self.convBlock3 = ConvBlock(128, 128, stride=2)
            self.convBlock4 = ConvBlock(128, 256, stride=1)
            self.convBlock5 = ConvBlock(256, 256, stride=2)
            self.convBlock6 = ConvBlock(256, 512, stride=1)
            self.convBlock7 = ConvBlock(512, 512, stride=2)

            self.avePool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化，指定输出（H，W）
            self.conv2 = nn.Conv2d(512, 1024, kernel_size=1)
            self.relu2 = nn.LeakyReLU()
            self.conv3 = nn.Conv2d(1024, 1, kernel_size=1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu1(x)

            x = self.convBlock1(x)
            x = self.convBlock2(x)
            x = self.convBlock3(x)
            x = self.convBlock4(x)
            x = self.convBlock5(x)
            x = self.convBlock6(x)
            x = self.convBlock7(x)

            x = self.avePool(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.conv3(x)
            x = self.sigmoid(x)

            return x