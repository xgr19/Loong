import torch
import torch.nn as nn
from operations import *
from genotypes import Genotype
import time
import random


input_tensor = torch.randn(1, 1, 50)
num_classes = 18
#geno = [[('dilg_conv_3_RELU', 0), ('skip_connect', 0), ('skip_connect', 0), ('max_pool_3', 3)], range(1, 5), [('conv_3_LeakyReLU', 0), ('avg_pool_3', 1), ('max_pool_3', 2), ('max_pool_3', 1)], range(1, 5), [('skip_connect', 0), ('avg_pool_5', 1), ('avg_pool_3', 1), ('dilg_conv_3_LeakyReLU', 0)], range(1, 5), [('avg_pool_3', 0), ('conv_3_RELU', 1), ('avg_pool_3', 0), ('max_pool_3', 2)], range(1, 5), [('avg_pool_3', 0), ('dilg_conv_3_RELU', 0), ('lp_pool_3', 1), ('lp_pool_3', 0)], range(1, 5), [('g_conv_3_ELU', 0), ('lp_pool_5', 1), ('avg_pool_3', 2), ('skip_connect', 1)], range(1, 5), [('max_pool_5', 0), ('max_pool_5', 1), ('skip_connect', 2), ('avg_pool_5', 2)], range(1, 5), [('g_conv_3_RELU', 0), ('lp_pool_5', 0), ('skip_connect', 2), ('g_conv_3_LeakyReLU', 3)], range(1, 5), [('lp_pool_3', 0), ('max_pool_5', 1), ('max_pool_3', 2), ('lp_pool_3', 0)], range(1, 5), [('max_pool_5', 0), ('avg_pool_3', 0), ('lp_pool_3', 0), ('lp_pool_5', 2)], range(1, 5)]
geno = None


PRIMITIVES = [
    # pool 
    'avg_pool_3',
    'avg_pool_5',
    'max_pool_3',
    'max_pool_5',
    'lp_pool_3',
    'lp_pool_5',
    # skip
    'skip_connect',
    # sep conv 
    'g_conv_3_RELU',
    'g_conv_3_ELU',
    'g_conv_3_LeakyReLU',
    'g_conv_5_RELU',
    'g_conv_5_ELU',
    'g_conv_5_LeakyReLU',
    # dil conv 
    'dilg_conv_3_RELU',
    'dilg_conv_3_ELU',
    'dilg_conv_3_LeakyReLU',
    'dilg_conv_5_RELU',
    'dilg_conv_5_ELU',
    'dilg_conv_5_LeakyReLU',
    # normal conv 
    'conv_3_RELU',
    'conv_3_ELU',
    'conv_3_LeakyReLU',
    'conv_5_RELU',
    'conv_5_ELU',
    'conv_5_LeakyReLU',
]


class Normal_Cell(nn.Module):

  def __init__(self, genotype, C_prev, C):
    super(Normal_Cell, self).__init__()

    self.preprocess = ReLUConvBN(C_prev, C, 1, 1, 0)

    op_names, indices = zip(*genotype.normal)
    concat = genotype.normal_concat

    self._compile(C, op_names, indices, concat)

  def _compile(self, C, op_names, indices, concat):
    assert len(op_names) == len(indices)
    self._steps = len(op_names)
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      op = OPS[name](C, 1, True)
      self._ops += [op]
    self._indices = indices

  def forward(self, s):
    s = self.preprocess(s)

    states = [s]
    for i in range(self._steps):
      h = states[self._indices[i]]
      op = self._ops[i]
      h = op(h)
      states += [h]

    return torch.cat([states[i] for i in self._concat], dim=1)


class Network(nn.Module):
    def __init__(self, cell_numbers, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        
        self._C = 64
        self._steps = steps
        self._multiplier = multiplier
        self._stem_multiplier = stem_multiplier
        C_curr = stem_multiplier * self._C
    
        self.stem = nn.Sequential(
            nn.Conv1d(1, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm1d(C_curr)
            )
        
        self.cells = nn.ModuleList()
        
        for i in range(cell_numbers):
            if i == 0:
                C_prev = self._stem_multiplier * self._C
            else:
                C_prev = self._multiplier * self._C
            self.cells += [Normal_Cell(Genotype(geno[i * 2], geno[i * 2 + 1]), C_prev, self._C)]
        C_prev = self._multiplier * self._C
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        
        
    def forward(self, x):
        h = self.stem(x)
        for i, cell in enumerate(self.cells):
            h = cell(h)
        h = self.global_pooling(h)
        h = h.view(h.size(0), -1)
        h = self.classifier(h)
        return h


if __name__ == '__main__':
    
    for num in range(10000):
        print('generating model: %d' % (num + 1))
        num_layers = random.randint(1, 10)
        
        geno = []
        for step in range(num_layers):
            operation_list = []
            for stage in range(4):
                operation_list.append((PRIMITIVES[random.randint(0, len(PRIMITIVES) - 1)], random.randint(0, stage)))
            geno.append(operation_list)
            geno.append(range(1, 5))
        
        #print(geno)
        model = Network(num_layers)
        
        #begin_time = time.time()
        #model(input_tensor)
        #print(model(input_tensor).shape)
        #print(time.time() - begin_time) # s, not ms 
        
        torch.onnx.export(model,  # model being run
                    input_tensor,  # model input (or a tuple for multiple inputs)
                    'result/model_num_' + str(num + 1) + '.onnx',  # where to save the model (can be a file or file-like object)
                    export_params=True,  # store the trained parameter weights inside the model file
                    opset_version=10,  # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names=['input'],  # the model's input names
                    output_names=['output']  # the model's output names)
                      )
