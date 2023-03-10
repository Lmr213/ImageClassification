import torch
from torch import nn
from torch.nn import functional as F


class ResBlk(nn.Module):
    """
    resnet block
    """
    def __init__(self, ch_in, ch_out, stride=1):
        """
        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        """
        :param x: [b, ch, h, w]
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # skip connection(用原来的x)
        # element-wise add: [b, ch_in, h, w] + [b, ch_out, h, w] need ch_in and ch_out same
        out = self.extra(x) + out
        out = F.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, num_class):
        super(ResNet, self).__init__()

        self.conv1 = nn.Sequential(    # 预处理层增加通道数
            nn.Conv2d(3, 16, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(16)
        )
        # following 4 blocks(h, w 的值也是一直变化的)
        # [b, 16, h, w] => [b, 32, h, w]
        self.blk1 = ResBlk(16, 32, stride=3)
        # [b, 32, h, w] => [b, 64, h, w]
        self.blk2 = ResBlk(32, 64, stride=3)
        # [b, 64, h, w] => [b, 128, h, w]
        self.blk3 = ResBlk(64, 128, stride=2)
        # [b, 128, h, w] => [b, 256, h, w]
        self.blk4 = ResBlk(128, 256, stride=2)

        # 要match好最后一个block的输出和最后线性层的输入
        self.outlayer = nn.Linear(256*3*3, num_class)


    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))

        # [b, 16, h, w] => [b, 256, h, w]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        # print(x.shape)
        # 加入全局平均池化层
        # 无论输入size多少，只会输出通道数个1*1的像素
        # [b, 256, h, w] => [b, 256, 1, 1]

        # x = F.adaptive_max_pool2d(x, [1, 1])
        x = x.view(x.size(0), -1)  # 展平操作
        x = self.outlayer(x)

        return x

def main():

    blk = ResBlk(64, 128)
    inp = torch.randn(2, 64, 224, 224)
    out = blk(inp)
    print('block:', out.shape)

    x = torch.randn(2, 3, 224, 224)
    model = ResNet(5)
    out = model(x)
    print('resnet:', out.shape)

    p = sum(map(lambda p: p.numel(), model.parameters()))
    print("parameters size:", p)

if __name__ == '__main__':
    main()
