import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvolutionalLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )
    def forward(self, x):
        return self.conv(x)

class ResidualLayer(nn.Module):
    def __init__(self, in_channels):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(
            ConvolutionalLayer(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0),
            ConvolutionalLayer(in_channels // 2,in_channels,kernel_size=3,stride=1,padding=1)
        )

    def forward(self, x):
        return x + self.resblock(x)

class DownSampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampleLayer, self).__init__()
        self.conv = nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        return self.conv(x)

class UpSampleLayer(nn.Module):
    def __init__(self):
        super(UpSampleLayer, self).__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode="nearest")

class ConvolutionalSetLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionalSetLayer, self).__init__()
        self.conv = nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            ConvolutionalLayer(out_channels, in_channels, kernel_size=3, stride=1, padding=1),
            ConvolutionalLayer(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            ConvolutionalLayer(out_channels, in_channels, kernel_size=3, stride=1, padding=1),
            ConvolutionalLayer(in_channels, out_channels, kernel_size=1,stride=1, padding=0)
        )

    def forward(self, x):
        return self.conv(x)

class Darknet53(nn.Module):
    def __init__(self,num_classes):
        super(Darknet53, self).__init__()

        self.num_classes = num_classes

        self.feature_52 = nn.Sequential(
            ConvolutionalLayer(3, 32, 3, 1, 1), # 3x3 卷积
            DownSampleLayer(32, 64),  # 下采样

            ResidualLayer(64),

            DownSampleLayer(64, 128),  # 3x3 ,strid=2,下采样

            ResidualLayer(128),
            ResidualLayer(128),

            DownSampleLayer(128, 256),  # 下采样

            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256)
        )  # 尺寸为52 x 52的特征图

        self.feature_26 = nn.Sequential(
            DownSampleLayer(256, 512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512)
        )  # 尺寸为26x26的特征图

        self.feature_13 = nn.Sequential(
            DownSampleLayer(512, 1024),  # 下采样
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024))

        self.convolset_13 = nn.Sequential(
            ConvolutionalSetLayer(1024,512)
        )

        self.convolset_26 = nn.Sequential(
            ConvolutionalSetLayer(768,256)
        )

        self.convolset_52 = nn.Sequential(
            ConvolutionalSetLayer(384,128)
        )

        self.detection_13 = nn.Sequential(
            ConvolutionalLayer(512, 1024, 3, 1, 1),
            nn.Conv2d(1024,(5 + self.num_classes)*3,1,1,0)
        ) # detection 13 x 13

        self.detection_26 = nn.Sequential(
            ConvolutionalLayer(256, 512, 3, 1, 1),
            nn.Conv2d(512,(5 + self.num_classes)*3,1,1,0)
        ) # detection 26 x 26

        self.detection_52 = nn.Sequential(
            ConvolutionalLayer(128, 256, 3, 1, 1),
            nn.Conv2d(256,(5 + self.num_classes)*3,1,1,0)
        ) # detection 52 x 52

        self.up_26 = nn.Sequential(
            ConvolutionalLayer(512, 256, 1, 1, 0),
            UpSampleLayer()
        ) # upsample 13->26

        self.up_52 = nn.Sequential(
            ConvolutionalLayer(256, 128, 1, 1, 0),
            UpSampleLayer()
        )  # upsample 26->52

    def forward(self, x):
        h_52 = self.feature_52(x)
        h_26 = self.feature_26(h_52)
        h_13 = self.feature_13(h_26)

        conval_13 = self.convolset_13(h_13)
        detection_13 = self.detection_13(conval_13)

        up_26 = self.up_26(conval_13)
        route_26 = torch.cat((up_26, h_26), dim=1)
        conval_26 = self.convolset_26(route_26)
        detection_26 = self.detection_26(conval_26)

        up_52 = self.up_52(conval_26)
        route_52 = torch.cat((up_52, h_52), dim=1)
        conval_52 = self.convolset_52(route_52)
        detection_52 = self.detection_52(conval_52)

        return detection_13, detection_26, detection_52


if __name__ == "__main__":

    anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
    dete_13_mask = [6, 7, 8]
    dete_26_mask = [3, 4, 5]
    dete_52_mask = [0,1,2]
