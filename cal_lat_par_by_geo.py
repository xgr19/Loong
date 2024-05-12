from genotypes import Genotype
import numpy as np
import time
import json

#target_device = 'gpu'

class Cal_Lat_Par_Cell(object):

  def __init__(self, cell_layer, genotype, channel, byte_len, target_device):
    self.latency = 0
    self.parameters = 0

    self.channel = channel
    self.byte_len = byte_len

    self.latency_table = None
    self.parameter_table = None
    #with open("latency_table_%s_%d.json" % (target_device, self.channel), 'r') as f:
    with open("latency_table_MNN_50.json", 'r') as f:
      self.latency_table = json.load(f)
    #with open("parameter_table_%d.json" % (self.channel), 'r') as f:
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
    
    # s is tuple type, shape[C_prev, length]
    # output shape[C, length]
    #s = self.preprocess(s)

    name = self.preprocess + '_' + str(s[0]) + 'x' + str(s[1])
    self.latency += self.latency_table[name]
    #self.parameters += self.parameter_table[self.preprocess]
    self.parameters += self.parameter_table[name]
    s[0] = self.channel

    states = [s]
    for i in range(self._steps):
        h = states[self._indices[i]]
        op = self._ops[i]
        #h = op(h)

        name = op + '_' + str(h[0]) + 'x' + str(h[1])
        self.latency += self.latency_table[name]
        #self.parameters += self.parameter_table[op]
        self.parameters += self.parameter_table[name]

        states += [h]

    # ignore the time of cat 
    return

'''
class Cal_Lat_Par_Network(object):

  def __init__(self, C, num_classes, layers, auxiliary, genotype, d_dim=channel):
    self._layers = layers
    self._auxiliary = auxiliary
    self.stem_multiplier = 3
    self.C = C
    self.d_dim = d_dim
    self.latency = 0
    self.parameters = 0

    C_curr = self.stem_multiplier * C

    self.embedding_and_stem = 'embedding_and_stem_' + str(byte_len) + '_' + str(d_dim) + '_' + str(C)

    C_prev, C_curr = C_curr, C
    self.cells = []

    for i in range(layers):
      if layers == 1:
        reduction = False
      elif i in [layers // 2]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(genotype, C_prev, C_curr, reduction)
      self.cells += [cell]
      C_prev = cell.multiplier * C_curr

    if layers >= 2:
        self.Avg_and_Linear = 'AdaptiveAvgPool1d_and_Linear_' + str(8 * C) + 'x129_' + str(num_classes)
    else:
        self.Avg_and_Linear = 'AdaptiveAvgPool1d_and_Linear_' + str(4 * C) + 'x258_' + str(num_classes)
    
    self.forward()


  def forward(self):
    s = [self.stem_multiplier * self.C, 258]
    self.latency += latency_table[self.embedding_and_stem]
    name = 'embedding_and_stem_' + str(self.d_dim) + '_' + str(self.C)
    self.parameters += parameter_table[name]

    for _, cell in enumerate(self.cells):
      t = cell.forward(s)
      s = t[0]

      self.latency += t[1]
      self.parameters += t[2]

    self.latency += latency_table[self.Avg_and_Linear]
    self.parameters += parameter_table[self.Avg_and_Linear]

    return
'''

if __name__ == '__main__':
    model = Cal_Lat_Par_Cell(cell_layer = 2,
                             genotype = Genotype([('g_conv_5_LeakyReLU', 0), ('g_conv_5_LeakyReLU', 1), ('g_conv_5_LeakyReLU', 2), ('g_conv_5_LeakyReLU', 2)], range(1, 5)),
                             channel = 64, byte_len = 44, target_device = 'gpu')

    print('latency = %.6fms' % (model.latency))
    print('parameters = %.6fMB' % (model.parameters))
