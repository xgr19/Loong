from genotypes import Genotype
import numpy as np
import time
import json

class Cal_Lat_Par_Cell(object):
  def __init__(self, cell_layer, genotype, channel, byte_len, target_device):
    self.latency = 0
    self.parameters = 0
    self.channel = channel
    self.byte_len = byte_len
    self.latency_table = None
    self.parameter_table = None
    with open("latency_table_MNN_50.json", 'r') as f:
      self.latency_table = json.load(f)
    with open("memory_table_MNN_50.json", 'r') as f:
      self.parameter_table = json.load(f)
    if cell_layer == 1:
      self.C_prev = 3 * self.channel
    else:
      self.C_prev = 4 * self.channel
    self.preprocess = 'ReLUConvBN_' + str(self.C_prev) + '_' + str(self.channel)
    op_names, indices = zip(*genotype.normal)
    concat = genotype.normal_concat
    self._compile(self.channel, op_names, indices, concat)
    self.forward([self.C_prev, self.byte_len])

  def _compile(self, C, op_names, indices, concat):
    assert len(op_names) == len(indices)
    self._steps = len(op_names)
    self._concat = concat
    self.multiplier = len(concat)
    self._ops = []
    for name, index in zip(op_names, indices):
      stride = 1
      op = str(name) + '_' + str(C) + '_' + str(stride)
      self._ops += [op]
    self._indices = indices

  def forward(self, s):
    name = self.preprocess + '_' + str(s[0]) + 'x' + str(s[1])
    self.latency += self.latency_table[name]
    self.parameters += self.parameter_table[name]
    s[0] = self.channel
    states = [s]
    for i in range(self._steps):
        h = states[self._indices[i]]
        op = self._ops[i]
        name = op + '_' + str(h[0]) + 'x' + str(h[1])
        self.latency += self.latency_table[name]
        self.parameters += self.parameter_table[name]
        states += [h]
    return

if __name__ == '__main__':
    model = Cal_Lat_Par_Cell(cell_layer = 2,
                             genotype = Genotype([('g_conv_5_LeakyReLU', 0), ('g_conv_5_LeakyReLU', 1), ('g_conv_5_LeakyReLU', 2), ('g_conv_5_LeakyReLU', 2)], range(1, 5)),
                             channel = 64, byte_len = 44, target_device = 'gpu')
    print('latency = %.6fms' % (model.latency))
    print('parameters = %.6fMB' % (model.parameters))
