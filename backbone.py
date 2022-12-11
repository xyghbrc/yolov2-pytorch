'''darknet-19_pytorch'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import time
import math

class CBL(nn.Module):
    def __init__(self, in_channels, out_channels, k_size, if_BN=True, if_Leaky=True):
        super(CBL, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                              stride=1, padding=k_size // 2, bias=False, dtype=torch.float32)
        self.BN = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, dtype=torch.float32)
        self.activ = nn.LeakyReLU(negative_slope=0.1)
        self.if_BN = if_BN
        self.if_Leaky = if_Leaky

    def forward(self, input):
        output = self.conv(input)
        if self.if_BN:
            output = self.BN(output)
        if self.if_Leaky:
            output = self.activ(output)
        return output

class three2one(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(three2one, self).__init__()
        self.layers = nn.Sequential(
            CBL(in_channels=in_channels, out_channels=out_channels, k_size=3),
            CBL(in_channels=out_channels, out_channels=in_channels, k_size=1)
        )

    def  forward(self, input):
        output = self.layers(input)
        return output

class reorg(nn.Module):
    def __init__(self, stride):
        super(reorg, self).__init__()
        self.stride = stride

    def forward(self, input):
        n, c, h, w = input.shape
        n_re, c_re, h_re, w_re = n, c // self.stride ** 2, h * self.stride, w * self.stride
        # 26 * 26 * 64 -> 52 * 52 * 16
        input = input.reshape(n_re, c_re, h_re, w_re)
        re_index = torch.arange(0, c, self.stride ** 2)
        for i in range(self.stride ** 2 - 1):
            re_index = torch.cat([re_index, torch.arange(0, c, self.stride ** 2) + i + 1])
        unfolded_input = F.unfold(input, self.stride, stride=self.stride)[:, re_index, :]
        return unfolded_input.contiguous().view(n, c * self.stride ** 2, h // self.stride, w // self.stride)


class darknet19(nn.Module):
    def __init__(self, net_cfg):
        super(darknet19, self).__init__()
        layers = []
        layer_curr = []
        for item in net_cfg:
            if item[0] == 'route':
                layers.append((item[-1], nn.Sequential(*layer_curr)))
                layer_curr = []
                continue
            if item[0] == 'CBL':
                layer_curr = self.make_CBL(layer_curr, item)
            elif item[0] == 'maxpool':
                layer_curr = self.make_maxpool(layer_curr, item)
            elif item[0] == 'three2one':
                layer_curr = self.make_three2one(layer_curr, item)
            elif item[0] == 'reorg':
                layer_curr = self.make_reorg(layer_curr, item)
        layers.append(('detection', nn.Sequential(*layer_curr)))
        layers = OrderedDict(layers)
        self.layers = nn.Sequential(layers)
        '''weight initial'''
        for m in self.modules():
            # is nn.Conv2d() instance ?
            if isinstance(m, nn.Conv2d):
                scale = math.sqrt(2 / (m.kernel_size[0] * m.kernel_size[1] * m.in_channels))
                m.weight.data = scale * m.weight.data.normal_(0, 1)

    def make_three2one(self, layers, three2one_cfg):
        for i in range(three2one_cfg[-1][-1]):
            layers.append(three2one(in_channels=three2one_cfg[-1][0], out_channels=three2one_cfg[-1][1]))
        return layers

    def make_CBL(self, layers, CBL_cfg):
        for item in CBL_cfg[1:]:
            if item[-1] == 'extra':
                layers.append(CBL(in_channels=item[0], out_channels=item[1], k_size=item[2], if_BN=item[-3], if_Leaky=item[-2]))
            else:
                layers.append(CBL(in_channels=item[0], out_channels=item[1], k_size=item[2]))
        return layers

    def make_maxpool(self, layers, maxpool_cfg):
        layers.append(nn.MaxPool2d(kernel_size=maxpool_cfg[-1][0], stride=maxpool_cfg[-1][1]))
        return layers

    def make_reorg(self, layers, reorg_cfg):
        layers.append(reorg(stride=reorg_cfg[-1][-1]))
        return layers

    def forward(self, input):
        output_16 = self.layers[0](input)
        output_24 = self.layers[1](output_16)
        output_27 = self.layers[2](output_16)
        detection = self.layers[3](torch.cat([output_27, output_24], dim=1))
        return detection

if __name__ == "__main__":
    '''CBL: (in_channels, out_channels, k_size)'''
    '''maxpool: (k_size, stride)'''
    '''three2one: (in_channels, out_channels, num_cycle)'''
    '''reorg: (stride)'''
    '''route: layers cut'''
    net_cfg = (
        ('CBL', (3, 32, 3)), ('maxpool', (2, 2)),
        ('CBL', (32, 64, 3)), ('maxpool', (2, 2)),
        ('three2one', (64, 128, 1)),
        ('CBL', (64, 128, 3)), ('maxpool', (2, 2)),
        ('three2one', (128, 256, 1)),
        ('CBL', (128, 256, 3)), ('maxpool', (2, 2)),
        ('three2one', (256, 512, 2)),
        ('CBL', (256, 512, 3)), ('route', '16'),       # route: 16
        ('maxpool', (2, 2)),
        ('three2one', (512, 1024, 2)),
        ('CBL', (512, 1024, 3), (1024, 1024, 3), (1024, 1024, 3)), ('route', '24'),    # route: 24
        ('CBL', (512, 64, 1)),
        ('reorg', (2, )), ('route', '27'),      # route: 27
        ('CBL',  (1280, 1024, 3), (1024, 125, 1, True, False, 'extra'))       # detection: 13*13*125
    )
    yolov2 = darknet19(net_cfg)
    input = torch.randn((2, 3, 416, 416), requires_grad=True)
    start_time = time.time()
    output = yolov2(input).sum()
    stop_time_for = time.time()
    output.backward()
    stop_time_back = time.time()
    print('forward_time: {0:.6f}\nbackward_time: {1:.6f}'.format(stop_time_for - start_time, stop_time_back - stop_time_for))

    # # 导出为onnx格式
    # torch.onnx.export(
    #     yolov2_net,
    #     input,
    #     'yolov2.onnx',
    # )