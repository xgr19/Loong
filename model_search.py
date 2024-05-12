import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype
from torchinfo import summary
import numpy as np
import time
import json
import math
from utils import drop_path
#from cal_lat_par_by_geo import Cal_Lat_Par_Network

init_channel = 64
target_device = 'gpu'

latency_table = 0
#with open("latency_table_%s_%d.json" % (target_device, init_channel), 'r') as f:
with open("latency_table_MNN_50.json", 'r') as f:
  latency_table = json.load(f)

parameter_table = 0
#with open("parameter_table_%d.json" % (init_channel), 'r') as f:
with open("memory_table_MNN_50.json", 'r') as f:
  parameter_table = json.load(f)


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
      #stride = 2 if reduction and index < 1 else 1
      stride = 1
      # only from node 0 need stride 
      op = OPS[name](C, stride, True)
      self._ops += [op]
    self._indices = indices

  def forward(self, s, drop_prob):
    s = self.preprocess(s)

    states = [s]
    for i in range(self._steps):
      h = states[self._indices[i]]
      op = self._ops[i]
      h = op(h)
      if self.training and drop_prob > 0.:
        if not isinstance(op, Identity):
          h = drop_path(h, drop_prob)
      states += [h]

    return torch.cat([states[i] for i in self._concat], dim=1)


class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    self._name = []
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm1d(C, affine=False))
      self._ops.append(op)
      name_pre = primitive + '_' + str(C) + '_' + str(stride)
      self._name.append(name_pre)

  def forward(self, x, weights):
    s = 0
    latency = torch.FloatTensor([0]).cuda()
    parameter = torch.FloatTensor([0]).cuda()
    for w, op, name in zip(weights, self._ops, self._name):
      s = s + w * op(x)
      #latency = latency + w * latency_table[name + '_' + str(x.shape[1]) + 'x' + str(x.shape[2])]
      #parameter = parameter + w * parameter_table[name]
      #parameter = parameter + w * parameter_table[name + '_' + str(x.shape[1]) + 'x' + str(x.shape[2])]

    return s, latency, parameter


class Super_Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev, C):
    super(Super_Cell, self).__init__()
    #self.reduction = reduction
    self.C_prev = C_prev
    self.C = C

    self.preprocess = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(1+i):
        #stride = 2 if reduction and j < 1 else 1
        stride = 1
        op = MixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, weights):
    latency = 0
    parameter = 0
    name_pre = 'ReLUConvBN_' + str(self.C_prev) + '_' + str(self.C)
    latency = latency + latency_table[name_pre + '_' + str(s0.shape[1]) + 'x' + str(s0.shape[2])]
    #parameter = parameter + parameter_table[name_pre]
    parameter = parameter + parameter_table[name_pre + '_' + str(s0.shape[1]) + 'x' + str(s0.shape[2])]
    s0 = self.preprocess(s0)

    states = [s0]
    offset = 0
    for i in range(self._steps):
      s = 0
      for j, h in enumerate(states):
        # dim_0 is result, dim_1 is expected_latency, dim_2 is expected_parameter 
        t = self._ops[offset+j](h, weights[offset+j])
        s = s + t[0]
        latency = latency + t[1]
        parameter = parameter + t[2]
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1), latency, parameter


class Network(nn.Module):

  def __init__(self, C, num_classes, criterion, total_args, steps=4, multiplier=4, stem_multiplier=3, d_dim=1):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = 1
    self.args = total_args
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self._stem_multiplier = stem_multiplier
    if self.args.have_embedding_layer == False:
      self.d_dim = d_dim
    else:
      self.d_dim = self._C
    # exp shows that use drop_path_prob may ruin the training process 
    self.drop_path_prob = 0.0
    self.final_geo = []
    self.is_all_cell_normal_cell = False

    self.all_normal_cell_latency = 0.0
    self.all_normal_cell_parameter = 0.0
    self.current_best_acc = 0.0

    self.use_loss_regulation = False
    self.relu_func = nn.ReLU(inplace=True)

    self.byteembedding = None
    if self.args.have_embedding_layer:
      self.byteembedding = nn.Embedding(num_embeddings=300, embedding_dim=self.d_dim)

    C_curr = stem_multiplier * C
    self.stem = nn.Sequential(
      nn.Conv1d(self.d_dim, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm1d(C_curr)
    )

    C_prev, C_curr = C_curr, C
    self.cells = nn.ModuleList()

    #reduction = False
    cell = Super_Cell(steps, multiplier, C_prev, C_curr)
    self.cells += [cell]
    C_prev = multiplier * C_curr

    self.global_pooling = nn.AdaptiveAvgPool1d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    name = None
    if self.args.have_embedding_layer == True:
      name = 'embedding_and_stem_' + str(self.args.byte_len) + '_' + str(self.d_dim) + '_' + str(self._C)
    else:
      name = 'stem_' + str(self.args.byte_len) + '_' + str(self._C)
    self.all_normal_cell_latency += latency_table[name]

    #if self.args.have_embedding_layer == True:
    #  name = 'embedding_and_stem_' + str(self.d_dim) + '_' + str(self._C)
    #else:
    #  name = 'stem_' + str(self._C)
    self.all_normal_cell_parameter += parameter_table[name]

    name = 'AdaptiveAvgPool1d_and_Linear_' + str(self._multiplier * self.args.init_channels) + 'x' + str(self.args.byte_len) + '_' + str(self._num_classes)
    self.all_normal_cell_latency += latency_table[name]
    self.all_normal_cell_parameter += parameter_table[name]
    #print(self.all_normal_cell_parameter)
    #exit(0)

    self._initialize_alphas()

    self.dec_rate = None
    self.reg_lambda = None
    self.init_latency = None
    self.init_parameter = None


  def change_top_supercell_to_normalcell(self):
    now_geno = self.genotype()

    '''
    if self._layers == 1:
      C_prev = self._stem_multiplier * self._C
      C = self._C
      reduction = False
    elif self._layers == 2:
      C_prev = self._multiplier * self._C
      C = self._C * 2
      reduction = True
    else:
      C_prev = self._multiplier * self._C * 2
      C = self._C * 2
      reduction = False
    '''
    if self._layers == 1:
      C_prev = self._stem_multiplier * self._C
    else:
      C_prev = self._multiplier * self._C
    C = self._C
    new_cell = Normal_Cell(now_geno, C_prev, C).cuda()
    self.cells[self._layers - 1] = new_cell
    return


  def add_top_one_supercell(self):

    self._initialize_alphas()
    self._layers += 1

    '''
    if self._layers == 2:
      self.classifier = nn.Linear(self._multiplier * self._C * 2, self._num_classes)
    
    if self._layers == 2:
      C_prev = self._multiplier * self._C
      C = self._C * 2
      reduction = True
    else:
      C_prev = self._multiplier * self._C * 2
      C = self._C * 2
      reduction = False
    '''
    
    C_prev = self._multiplier * self._C
    C = self._C
    new_cell = Super_Cell(self._steps, self._multiplier, C_prev, C).cuda()
    self.cells += [new_cell]
    return


  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    expected_latency = 0
    expected_parameter = 0

    #name = 'embedding_and_stem_' + '40_' + str(self.d_dim) + '_' + str(self._C)
    #expected_latency = expected_latency + latency_table[name]
    #name = 'embedding_and_stem_' + str(self.d_dim) + '_' + str(self._C)
    #expected_parameter = expected_parameter + parameter_table[name]
    if self.args.have_embedding_layer:
      input = self.byteembedding(input)
      input = input.transpose(-2, -1)

    s = self.stem(input)

    for i, cell in enumerate(self.cells):
      if i < self._layers - 1 or self.is_all_cell_normal_cell == True: # normal_cell 
        s = cell(s, self.drop_path_prob)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
        s, latency, parameter = cell(s, weights)
        expected_latency = expected_latency + latency
        expected_parameter = expected_parameter + parameter

    #name = 'AdaptiveAvgPool1d_and_Linear_' + str(s.shape[1]) + 'x' + str(s.shape[2]) + '_' + str(self._num_classes)
    #expected_latency = expected_latency + latency_table[name]
    #expected_parameter = expected_parameter + parameter_table[name]

    out = self.global_pooling(s)
    logits = self.classifier(out.view(out.size(0), -1))

    return logits, expected_latency, expected_parameter

  def _loss(self, input, target):
    logits, expected_latency, expected_parameter = self(input)
    ce_loss = self._criterion(logits, target)

    if self.use_loss_regulation == False:
      return ce_loss
    
    reg_loss_type = 'add#linear'
    # we prefer loss type add and linear 

    if reg_loss_type == 'mul#log':
      ref_value_latency = 190
      ref_value_parameter = 0.60
      alpha = 1
      beta = 0.6

      reg_loss_latency = (torch.log(expected_latency) / math.log(ref_value_latency)) ** beta
      reg_loss_parameter = (torch.log(expected_parameter * 100) / math.log(ref_value_parameter * 100)) ** beta
      reg_loss_latency = max(reg_loss_latency, 1.0)
      reg_loss_parameter = max(reg_loss_parameter, 1.0)
      #print(expected_latency.item(), reg_loss_latency.item())

      return alpha * ce_loss * reg_loss_latency * reg_loss_parameter
    
    else: # loss type add and linear 
      ref_value_latency = self.init_latency * self.dec_rate
      ref_value_parameter = self.init_parameter * self.dec_rate

      reg_loss_latency = self.reg_lambda * (expected_latency - ref_value_latency) / ref_value_latency
      reg_loss_parameter = self.reg_lambda * (expected_parameter - ref_value_parameter) / ref_value_parameter

      if self.args.latency_restrict == 0.0 and self.args.parameter_restrict == 0.0:
        return ce_loss
      
      if self.args.latency_restrict > 0.0 and self.args.parameter_restrict == 0.0:
        return ce_loss + self.relu_func(reg_loss_latency)

      if self.args.latency_restrict == 0.0 and self.args.parameter_restrict > 0.0:
        return ce_loss + self.relu_func(reg_loss_parameter)
      
      if self.args.latency_restrict > 0.0 and self.args.parameter_restrict > 0.0:
        return ce_loss + self.relu_func(reg_loss_latency) + self.relu_func(reg_loss_parameter)


  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(1 + i))
    num_ops = len(PRIMITIVES)

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    #self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      #self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights):
      gene = []
      n = 1
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 1), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:1]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k_best is None or W[j][k] > W[j][k_best]:
              k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    #gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(1 + self._steps - self._multiplier, self._steps + 1)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat
      #reduce=gene_reduce, reduce_concat=concat
    )
    return genotype



if __name__ == '__main__':

  test_device = 'gpu'
  # device type must be gpu 

  criterion = nn.CrossEntropyLoss()

  if test_device == 'cpu':
    criterion = criterion.cpu()
  else:
    criterion = criterion.cuda()

  model = Network(C=32, num_classes=10, layers=1, criterion=criterion)

  if test_device == 'cpu':
    model = model.cpu()
  else:
    model = model.cuda()

  model.eval()

  with torch.no_grad():
    input = torch.randn(256, 1, 40)

    if test_device == 'cpu':
      input = input.cpu()
    else:
      input = input.cuda()

    for i in range(20):
      start_time = time.time()
      model(input)
      print('%.3fms' % ((time.time() - start_time) * 1e3))

