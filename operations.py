import torch
import torch.nn as nn

OPS = {
  # three pool, avgpool, maxpool, lppool 
  'avg_pool_3' : lambda C, stride, affine: nn.AvgPool1d(3, stride=stride, padding=1, count_include_pad=False),
  'avg_pool_5' : lambda C, stride, affine: nn.AvgPool1d(5, stride=stride, padding=2, count_include_pad=False),
  'avg_pool_7' : lambda C, stride, affine: nn.AvgPool1d(7, stride=stride, padding=3, count_include_pad=False),
  'max_pool_3' : lambda C, stride, affine: nn.MaxPool1d(3, stride=stride, padding=1),
  'max_pool_5' : lambda C, stride, affine: nn.MaxPool1d(5, stride=stride, padding=2),
  'max_pool_7' : lambda C, stride, affine: nn.MaxPool1d(7, stride=stride, padding=3),
  'lp_pool_3' : lambda C, stride, affine: nn.Sequential(
    nn.ZeroPad2d(padding=(1, 1, 0, 0)),
    nn.LPPool1d(norm_type=2, kernel_size=3, stride=stride)
  ),
  'lp_pool_5' : lambda C, stride, affine: nn.Sequential(
    nn.ZeroPad2d(padding=(2, 2, 0, 0)),
    nn.LPPool1d(norm_type=2, kernel_size=5, stride=stride)
  ),
  'lp_pool_7' : lambda C, stride, affine: nn.Sequential(
    nn.ZeroPad2d(padding=(3, 3, 0, 0)),
    nn.LPPool1d(norm_type=2, kernel_size=7, stride=stride)
  ),
  
  # skip
  'skip_connect' : lambda C, stride, affine: Identity(),
  
  # sep conv 
  #'sep_3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  #'sep_5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  'g_conv_3_RELU' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv1d(C, C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False),
    nn.BatchNorm1d(C, affine=affine)
    ),
  
  'g_conv_3_ELU' : lambda C, stride, affine: nn.Sequential(
    nn.ELU(inplace=False),
    nn.Conv1d(C, C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False),
    nn.BatchNorm1d(C, affine=affine)
    ),
  
  'g_conv_3_LeakyReLU' : lambda C, stride, affine: nn.Sequential(
    nn.LeakyReLU(inplace=False),
    nn.Conv1d(C, C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False),
    nn.BatchNorm1d(C, affine=affine)
    ),
  
  'g_conv_5_RELU' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv1d(C, C, kernel_size=5, stride=stride, padding=2, groups=C, bias=False),
    nn.BatchNorm1d(C, affine=affine)
    ),
  
  'g_conv_5_ELU' : lambda C, stride, affine: nn.Sequential(
    nn.ELU(inplace=False),
    nn.Conv1d(C, C, kernel_size=5, stride=stride, padding=2, groups=C, bias=False),
    nn.BatchNorm1d(C, affine=affine)
    ),
  
  'g_conv_5_LeakyReLU' : lambda C, stride, affine: nn.Sequential(
    nn.LeakyReLU(inplace=False),
    nn.Conv1d(C, C, kernel_size=5, stride=stride, padding=2, groups=C, bias=False),
    nn.BatchNorm1d(C, affine=affine)
    ),
  
  'g_conv_7_RELU' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv1d(C, C, kernel_size=7, stride=stride, padding=3, groups=C, bias=False),
    nn.BatchNorm1d(C, affine=affine)
    ),
  
  'g_conv_7_ELU' : lambda C, stride, affine: nn.Sequential(
    nn.ELU(inplace=False),
    nn.Conv1d(C, C, kernel_size=7, stride=stride, padding=3, groups=C, bias=False),
    nn.BatchNorm1d(C, affine=affine)
    ),
  
  'g_conv_7_LeakyReLU' : lambda C, stride, affine: nn.Sequential(
    nn.LeakyReLU(inplace=False),
    nn.Conv1d(C, C, kernel_size=7, stride=stride, padding=3, groups=C, bias=False),
    nn.BatchNorm1d(C, affine=affine)
    ),
  
  # dil conv 
  #'dil_3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  #'dil_5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
  'dilg_conv_3_RELU' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv1d(C, C, kernel_size=3, stride=stride, padding=2, dilation=2, groups=C, bias=False),
    nn.BatchNorm1d(C, affine=affine)
    ),
  
  'dilg_conv_3_ELU' : lambda C, stride, affine: nn.Sequential(
    nn.ELU(inplace=False),
    nn.Conv1d(C, C, kernel_size=3, stride=stride, padding=2, dilation=2, groups=C, bias=False),
    nn.BatchNorm1d(C, affine=affine)
    ),
  
  'dilg_conv_3_LeakyReLU' : lambda C, stride, affine: nn.Sequential(
    nn.LeakyReLU(inplace=False),
    nn.Conv1d(C, C, kernel_size=3, stride=stride, padding=2, dilation=2, groups=C, bias=False),
    nn.BatchNorm1d(C, affine=affine)
    ),
  
  'dilg_conv_5_RELU' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv1d(C, C, kernel_size=5, stride=stride, padding=4, dilation=2, groups=C, bias=False),
    nn.BatchNorm1d(C, affine=affine)
    ),
  
  'dilg_conv_5_ELU' : lambda C, stride, affine: nn.Sequential(
    nn.ELU(inplace=False),
    nn.Conv1d(C, C, kernel_size=5, stride=stride, padding=4, dilation=2, groups=C, bias=False),
    nn.BatchNorm1d(C, affine=affine)
    ),
  
  'dilg_conv_5_LeakyReLU' : lambda C, stride, affine: nn.Sequential(
    nn.LeakyReLU(inplace=False),
    nn.Conv1d(C, C, kernel_size=5, stride=stride, padding=4, dilation=2, groups=C, bias=False),
    nn.BatchNorm1d(C, affine=affine)
    ),
  
  'dilg_conv_7_RELU' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv1d(C, C, kernel_size=7, stride=stride, padding=6, dilation=2, groups=C, bias=False),
    nn.BatchNorm1d(C, affine=affine)
    ),
  
  'dilg_conv_7_ELU' : lambda C, stride, affine: nn.Sequential(
    nn.ELU(inplace=False),
    nn.Conv1d(C, C, kernel_size=7, stride=stride, padding=6, dilation=2, groups=C, bias=False),
    nn.BatchNorm1d(C, affine=affine)
    ),
  
  'dilg_conv_7_LeakyReLU' : lambda C, stride, affine: nn.Sequential(
    nn.LeakyReLU(inplace=False),
    nn.Conv1d(C, C, kernel_size=7, stride=stride, padding=6, dilation=2, groups=C, bias=False),
    nn.BatchNorm1d(C, affine=affine)
    ),
  
  # normal conv, three activation function "RELU, ELU, LeakyReLU", kernel size[3, 5] 
  'conv_3_RELU' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv1d(C, C, 3, stride=1, padding=1, bias=False),
    nn.BatchNorm1d(C, affine=affine)
    ),
  
  'conv_3_ELU' : lambda C, stride, affine: nn.Sequential(
    nn.ELU(inplace=False),
    nn.Conv1d(C, C, 3, stride=1, padding=1, bias=False),
    nn.BatchNorm1d(C, affine=affine)
    ),
  
  'conv_3_LeakyReLU' : lambda C, stride, affine: nn.Sequential(
    nn.LeakyReLU(inplace=False),
    nn.Conv1d(C, C, 3, stride=1, padding=1, bias=False),
    nn.BatchNorm1d(C, affine=affine)
    ),
  
  'conv_5_RELU' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv1d(C, C, 5, stride=1, padding=2, bias=False),
    nn.BatchNorm1d(C, affine=affine)
    ),
  
  'conv_5_ELU' : lambda C, stride, affine: nn.Sequential(
    nn.ELU(inplace=False),
    nn.Conv1d(C, C, 5, stride=1, padding=2, bias=False),
    nn.BatchNorm1d(C, affine=affine)
    ),
  
  'conv_5_LeakyReLU' : lambda C, stride, affine: nn.Sequential(
    nn.LeakyReLU(inplace=False),
    nn.Conv1d(C, C, 5, stride=1, padding=2, bias=False),
    nn.BatchNorm1d(C, affine=affine)
    ),
  
  'conv_7_RELU' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv1d(C, C, 7, stride=1, padding=3, bias=False),
    nn.BatchNorm1d(C, affine=affine)
    ),
  
  'conv_7_ELU' : lambda C, stride, affine: nn.Sequential(
    nn.ELU(inplace=False),
    nn.Conv1d(C, C, 7, stride=1, padding=3, bias=False),
    nn.BatchNorm1d(C, affine=affine)
    ),
  
  'conv_7_LeakyReLU' : lambda C, stride, affine: nn.Sequential(
    nn.LeakyReLU(inplace=False),
    nn.Conv1d(C, C, 7, stride=1, padding=3, bias=False),
    nn.BatchNorm1d(C, affine=affine)
    ),
}

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv1d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm1d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)

class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv1d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv1d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm1d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv1d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv1d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm1d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv1d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv1d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm1d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv1d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv1d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    #self.conv_1 = nn.Conv1d(C_in, C_out, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm1d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:])], dim=1)
    #out = self.conv_1(x)
    out = self.bn(out)
    return out