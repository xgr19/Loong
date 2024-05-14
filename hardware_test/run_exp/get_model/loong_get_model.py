import torch
import torch.nn as nn
from operations import *
import time


dataset = 'cifar10'
#dataset = 'lor_cifar10'

test_batch_size = 1

if dataset == 'cifar10':
  x = torch.randn(test_batch_size, 3, 32, 32)
  num_classes = 10
  num_layers = 15
  geno = [['conv_1_RELU', 'conv_3_RELU', 'conv_3_RELU', 'conv_1_RELU', 'conv_3_RELU', 'conv_3_RELU'],
          ['avg_pool_3', 'avg_pool_3', 'conv_3_RELU', 'skip_connect', 'conv_1_RELU', 'conv_3_RELU'],
          ['avg_pool_3', 'avg_pool_3', 'conv_3_RELU', 'skip_connect', 'skip_connect', 'avg_pool_3'],
          ['skip_connect', 'avg_pool_3', 'avg_pool_3', 'skip_connect', 'avg_pool_3', 'skip_connect'],
          ['skip_connect', 'skip_connect', 'skip_connect', 'avg_pool_3', 'avg_pool_3', 'skip_connect'],
          ['conv_3_RELU', 'conv_3_RELU', 'conv_3_RELU', 'conv_3_RELU', 'conv_3_RELU', 'conv_3_RELU'],
          ['avg_pool_3', 'skip_connect', 'conv_3_RELU', 'skip_connect', 'skip_connect', 'skip_connect'],
          ['skip_connect', 'avg_pool_3', 'avg_pool_3', 'skip_connect', 'skip_connect', 'avg_pool_3'],
          ['conv_3_RELU', 'conv_3_RELU', 'conv_3_RELU', 'conv_1_RELU', 'conv_1_RELU', 'conv_3_RELU'],
          ['skip_connect', 'skip_connect', 'skip_connect', 'avg_pool_3', 'avg_pool_3', 'skip_connect'],
          ['conv_1_RELU', 'conv_3_RELU', 'conv_3_RELU', 'conv_1_RELU', 'conv_3_RELU', 'conv_3_RELU'],
          ['none', 'conv_1_RELU', 'none', 'none', 'none', 'conv_1_RELU'],
          ['skip_connect', 'avg_pool_3', 'skip_connect', 'skip_connect', 'skip_connect', 'avg_pool_3'],
          ['none', 'conv_1_RELU', 'none', 'conv_1_RELU', 'none', 'none'],
          ['skip_connect', 'avg_pool_3', 'none', 'skip_connect', 'skip_connect', 'none']]
  
elif dataset == 'lor_cifar10':
  x = torch.randn(test_batch_size, 3, 32, 32)
  num_classes = 10
  num_layers = 15
  geno = [['none', 'skip_connect', 'skip_connect', 'skip_connect', 'conv_3_RELU', 'conv_3_RELU'],
          ['avg_pool_3', 'avg_pool_3', 'conv_1_RELU', 'skip_connect', 'none', 'skip_connect'],
          ['conv_3_RELU', 'conv_3_RELU', 'skip_connect', 'conv_3_RELU', 'skip_connect', 'none'],
          ['avg_pool_3', 'skip_connect', 'skip_connect', 'skip_connect', 'none', 'conv_1_RELU'],
          ['conv_1_RELU', 'none', 'conv_1_RELU', 'conv_3_RELU', 'skip_connect', 'conv_1_RELU'],
          ['avg_pool_3', 'skip_connect', 'skip_connect', 'skip_connect', 'skip_connect', 'skip_connect'],
          ['avg_pool_3', 'skip_connect', 'avg_pool_3', 'conv_1_RELU', 'avg_pool_3', 'skip_connect'],
          ['none', 'conv_3_RELU', 'skip_connect', 'avg_pool_3', 'none', 'none'],
          ['none', 'none', 'conv_1_RELU', 'avg_pool_3', 'conv_1_RELU', 'none'],
          ['avg_pool_3', 'avg_pool_3', 'none', 'avg_pool_3', 'skip_connect', 'conv_3_RELU'],
          ['conv_3_RELU', 'conv_1_RELU', 'none', 'conv_3_RELU', 'conv_1_RELU', 'none'],
          ['avg_pool_3', 'avg_pool_3', 'skip_connect', 'avg_pool_3', 'none', 'none'],
          ['skip_connect', 'skip_connect', 'none', 'avg_pool_3', 'skip_connect', 'avg_pool_3'],
          ['avg_pool_3', 'none', 'conv_1_RELU', 'none', 'conv_1_RELU', 'none'],
          ['skip_connect', 'avg_pool_3', 'avg_pool_3', 'skip_connect', 'skip_connect', 'avg_pool_3']]


class ReLUConvBN(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        dilation,
        affine,
        track_running_stats=True,
    ):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_out,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=not affine,
            ),
            nn.BatchNorm2d(
                C_out, affine=affine, track_running_stats=track_running_stats
            ),
        )

    def forward(self, x):
        return self.op(x)


class Residual_block(nn.Module):
    def __init__(self, inplanes, planes, stride, affine=True, track_running_stats=True):
        super(Residual_block, self).__init__()
        assert stride == 1 or stride == 2, "invalid stride {:}".format(stride)
        self.conv_a = ReLUConvBN(
            inplanes, planes, 3, stride, 1, 1, affine, track_running_stats
        )
        self.conv_b = ReLUConvBN(
            planes, planes, 3, 1, 1, 1, affine, track_running_stats
        )
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(
                    inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False
                ),
            )
        elif inplanes != planes:
            self.downsample = ReLUConvBN(
                inplanes, planes, 1, 1, 0, 1, affine, track_running_stats
            )
        else:
            self.downsample = None
        self.in_dim = inplanes
        self.out_dim = planes
        self.stride = stride
        self.num_conv = 2

    def extra_repr(self):
        string = "{name}(inC={in_dim}, outC={out_dim}, stride={stride})".format(
            name=self.__class__.__name__, **self.__dict__
        )
        return string

    def forward(self, inputs):

        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        return residual + basicblock


class Normal_Cell(nn.Module):

  def __init__(self, edge_choice, C):
    super(Normal_Cell, self).__init__()
    stride = 1
    self.op_1 = OPS[edge_choice[0]](C, stride, False)
    self.op_2 = OPS[edge_choice[1]](C, stride, False)
    self.op_3 = OPS[edge_choice[2]](C, stride, False)
    self.op_4 = OPS[edge_choice[3]](C, stride, False)
    self.op_5 = OPS[edge_choice[4]](C, stride, False)
    self.op_6 = OPS[edge_choice[5]](C, stride, False)

  def forward(self, input):

    node_1 = input
    meta_1 = self.op_1(node_1)
    node_2 = meta_1
    meta_2 = self.op_2(node_1)
    meta_3 = self.op_3(node_2)
    node_3 = meta_2 + meta_3
    meta_4 = self.op_4(node_1)
    meta_5 = self.op_5(node_2)
    meta_6 = self.op_6(node_3)
    node_4 = meta_4 + meta_5 + meta_6

    return node_4


class Network(nn.Module):
    def __init__(self, cell_numbers):
        super(Network, self).__init__()
        
        self._C = 16
    
        self.stem = nn.Sequential(
          nn.Conv2d(3, self._C, 3, padding=1, bias=False),
          nn.BatchNorm2d(self._C)
          )
        
        self.cells = nn.ModuleList()
        
        for i in range(cell_numbers):
          if i < 5:
            self.cells += [Normal_Cell(geno[i], self._C)]
          elif i == 5:
            self.cells += [Residual_block(self._C, self._C * 2, 2)]
            self.cells += [Normal_Cell(geno[i], self._C * 2)]
          elif i < 10:
            self.cells += [Normal_Cell(geno[i], self._C * 2)]
          elif i == 10:
            self.cells += [Residual_block(self._C * 2, self._C * 4, 2)]
            self.cells += [Normal_Cell(geno[i], self._C * 4)]
          else:
            self.cells += [Normal_Cell(geno[i], self._C * 4)]

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        
        if cell_numbers <= 5:
          self.classifier = nn.Linear(self._C, num_classes)
        elif cell_numbers <= 10:
          self.classifier = nn.Linear(self._C * 2, num_classes)
        else:
          self.classifier = nn.Linear(self._C * 4, num_classes)
        
        
    def forward(self, x):
        h = self.stem(x)
        for _, cell in enumerate(self.cells):
            h = cell(h)
        h = self.global_pooling(h)
        h = h.view(h.size(0), -1)
        h = self.classifier(h)
        return h


if __name__ == '__main__':
    
    model = Network(num_layers)
        
    begin_time = time.time()
    model(x)
    print(model(x).shape)
    print(time.time() - begin_time) # s, not ms 
