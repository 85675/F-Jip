import torch.nn as nn

class PMG(nn.Module):
    def __init__(self, model, feature_size, classes_num):
        super(PMG, self).__init__()

        self.features = model
        self.max3 = nn.MaxPool2d((14, 14), stride=(14, 14))
        self.num_ftrs = 2048 * 1 * 1
        self.classifier1 = nn.Sequential(
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
        )
        self.classifier3 = nn.Sequential(
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
        )

    def forward(self, x):
        xf1, xf2, xf3, xf4, xf5 = self.features(x)
        #
        xl1 = self.max3(xf5)
        xl1 = xl1.view(xl1.size(0), -1)
        xc1 = self.classifier1(xl1)

        xl3 = self.max3(xf5)
        xl3 = xl3.view(xl3.size(0), -1)
        xc3 = self.classifier3(xl3)

        return xc1,xc3


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
