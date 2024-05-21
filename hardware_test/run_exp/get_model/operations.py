import torch.nn as nn

OPS = {
  'avg_pool_3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'conv_1_RELU' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, 1, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(C, affine=affine)
    ),
  'conv_3_RELU' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, 3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(C, affine=affine)
    ),
  'skip_connect' : lambda C, stride, affine: Identity(),
  'none' : lambda C, stride, affine: Zero(),
}

class Identity(nn.Module):
  def __init__(self):
    super(Identity, self).__init__()
  def forward(self, x):
    return x

class Zero(nn.Module):
  def __init__(self):
    super(Zero, self).__init__()
  def forward(self, x):
    return x.mul(0.)
