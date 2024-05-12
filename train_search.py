import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network
from architect import Architect

import pickle
from cal_lat_par_by_geo import Cal_Lat_Par_Cell

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--expected_value_decrease_rate', type=float, default=0.9, help='expected value decrease rate')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--use_loss_regulation', type=bool, default=True, help='wheather use expected latency')
parser.add_argument('--num_classes', type=int, default=6, help='number of classes')
parser.add_argument('--byte_len', type=int, default=50, help='length of input(bytes)')
parser.add_argument('--have_embedding_layer', type=bool, default=False, help='if need embedding layer')
parser.add_argument('--latency_restrict', type=float, default=0.0, help='final model latency cannot beyond')
parser.add_argument('--parameter_restrict', type=float, default=0.0, help='final model parameter cannot beyond')
parser.add_argument('--test_latency_device', type=str, default='gpu', help='choose which device of latency table')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--add_cell_epochs', type=int, default=10, help='add a cell per epochs')
parser.add_argument('--max_layers', type=int, default=10, help='final net have max numbers of layers')
parser.add_argument('--mid_eval_epochs', type=int, default=30, help='mid eval epochs before add a new supercell')
parser.add_argument('--final_training_epochs', type=int, default=30, help='final net training epochs')
parser.add_argument('--init_channels', type=int, default=64, help='num of init channels')
parser.add_argument('--super_cell_early_stopping', type=int, default=2, help='early stopping if acc not improve')
parser.add_argument('--skip_connect_early_stopping', type=int, default=1, help='early stopping if skip_connect appear')
parser.add_argument('--reg_lambda', type=float, default=3e-2, help='reg lambda in the loss')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2022, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--max_search_training_samples', type=int, default=200000, help='max training samples when searching')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=2e-4, help='learning rate for arch encoding') # default is 3e-4 
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()
# 5841M 
assert args.latency_restrict == 0.0 or args.latency_restrict >= 1.5 
assert args.parameter_restrict >= 0.0
### epoc shouldn't be set very large, or will cause cosine scheduler lose effect(lr not change) 

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


protocols = ['firefox-bin', 'Safari', 'Mail', 'amule', 'Transmission', 'bittorrent.exe']
CLASSES = args.num_classes


class Dataset(torch.utils.data.Dataset):
	"""docstring for Dataset"""
	def __init__(self, x, label):
		super(Dataset, self).__init__()
		self.x = np.reshape(x, (-1, 1, args.byte_len))
		self.label = label

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		return self.x[idx], self.label[idx]

def load_epoch_data(flow_dict, train='train'):
	flow_dict = flow_dict[train]
	x, label = [], []

	for p in protocols:
		pkts = flow_dict[p]
		for byte, pos in pkts:
			x.append(byte)
			label.append(protocols.index(p))

	return np.array(x), np.array(label)[:, np.newaxis]

def paired_collate_fn(insts):
  x, label = list(zip(*insts))
  return torch.FloatTensor(np.array(x)), torch.LongTensor(np.array(label)).contiguous().view(-1)


class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)
  #logging.info("expected_value_decrease_rate = %f", args.expected_value_decrease_rate)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, CLASSES, criterion, total_args = args)
  model = model.cuda()
  test_latency_tensor = torch.randn(args.batch_size, 1, args.byte_len)
  _, init_latency, init_parameter = model(test_latency_tensor.cuda())
  model.init_latency = init_latency.item()
  model.init_parameter = init_parameter.item()
  # model.init_latency should be float type, or it will be a trainable tensor and cause errors:
  # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
  model.dec_rate = args.expected_value_decrease_rate
  model.reg_lambda = args.reg_lambda
  logging.info("init_latency = %fms, init_parameter = %fMB", model.init_latency, model.init_parameter)
  logging.info("ref_latency = %fms, ref_parameter = %fMB", model.init_latency * model.dec_rate, model.init_parameter * model.dec_rate)
  logging.info("supernet param size = %fMB", utils.count_parameters_in_MB(model))


  flow_dict = None
  with open('unibs_flows_0_noip_fold.pkl', 'rb') as f:
    flow_dict = pickle.load(f)


  train_x, train_label = load_epoch_data(flow_dict, 'train')
  train_data = Dataset(x=train_x, label=train_label)

  num_train = len(train_data)
  indices = list(np.random.choice(num_train, size=num_train, replace=False))
  split = min(int(np.floor(args.train_portion * num_train)), args.max_search_training_samples // 2)

  train_queue = torch.utils.data.DataLoader(
      train_data, collate_fn=paired_collate_fn,
      batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=0)
      # num_workers=2 will cause error and slow down the running

  valid_queue = torch.utils.data.DataLoader(
      train_data, collate_fn=paired_collate_fn,
      batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:min(num_train, split * 2)]),
      pin_memory=True, num_workers=0)

  now_layers = 1
  global_best_layers = 0
  
  global_training_early_stopping = 0

  while now_layers <= args.max_layers:

    optimizer = torch.optim.SGD(
                model.parameters(),
                args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                  optimizer, float(args.add_cell_epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)


    early_stopping = 0
    last_valid_acc = 0
    for epoch in range(args.add_cell_epochs):
      logging.info('epoch %d layers %d lr %f', epoch + 1, min(now_layers, args.max_layers), scheduler.get_last_lr()[0])

      
      model.use_loss_regulation = False 
      if args.use_loss_regulation and epoch >= 3:
        model.use_loss_regulation = True

      train_acc, _ = train(epoch, train_queue, valid_queue, model, architect, criterion, optimizer, scheduler.get_last_lr()[0])
      logging.info('train_acc %f', train_acc)
      scheduler.step()

      # validation
      valid_acc, _ = infer(valid_queue, model, criterion)
      logging.info('valid_acc %f', valid_acc)

      if args.use_loss_regulation == True:
        skip_num = 0
        for i in model.genotype().normal: # type model.genotype().normal is list, member is tuple 
          if i[0] == 'skip_connect' or 'pool' in i[0]:
          #if i[0] == 'skip_connect':
            skip_num += 1

        if skip_num > 0.5:
          logging.info('skip_or_pool_num = %d, early stopping', skip_num)
          break


      if valid_acc > last_valid_acc:
        last_valid_acc = valid_acc
        early_stopping = 0
      else:
        early_stopping += 1
      
      if early_stopping == args.super_cell_early_stopping:
        logging.info('due to valid_acc not improve in %d epochs, early stopping', args.super_cell_early_stopping)
        break


      #utils.save(model, os.path.join(args.save, 'weights.pt'))

    model.final_geo += model.genotype()
    logging.info('model.final_geo = %s', model.final_geo)

    #cal_model = Cal_Lat_Par_Cell(cell_layer = model._layers, genotype = model.genotype(), 
    #                             channel = args.init_channels, byte_len = args.byte_len, 
    #                             target_device = args.test_latency_device)
    
    #model.all_normal_cell_latency += cal_model.latency
    #model.all_normal_cell_parameter += cal_model.parameters
    #logging.info('normal_cell_latency = %fms, normal_cell_parameter = %fMB', 
    #              model.all_normal_cell_latency, model.all_normal_cell_parameter)

    model.change_top_supercell_to_normalcell()
    model = model.cuda()
    logging.info('test model parameter = %fMB', utils.count_parameters_in_MB(model))
    model.is_all_cell_normal_cell = True

    logging.info('latency_restrict = %fms, model.all_normal_cell_latency = %fms', args.latency_restrict, model.all_normal_cell_latency)
    
    # will never run into this condition in this latest code 
    #if args.latency_restrict > 0 and args.latency_restrict < model.all_normal_cell_latency:
    #  # remove the top normal cell 
    #  model.cells = model.cells[:-1]
    #  model._layers = len(model.cells)
    #  logging.info('due to latency restrict, stop adding cells, final layers = %d', model._layers)
    #  model.all_normal_cell_latency -= last_cell_latency
    #  logging.info('final network latency = %fms', model.all_normal_cell_latency)
    #  break

    best_acc = mid_training_for_eval(model, flow_dict, args.mid_eval_epochs, now_layers)

    logging.info('now layers = %d', model._layers)
    logging.info('global_valid_best_acc = %f, new valid_best_acc = %f', model.current_best_acc, best_acc)
    
    if best_acc > model.current_best_acc:
      model.current_best_acc = best_acc
      global_best_layers = model._layers
      global_training_early_stopping = 0
    else:
      global_training_early_stopping += 1
    
    logging.info('current_valid_best_acc = %f , global_valid_best_layers = %d', model.current_best_acc, global_best_layers)
    if global_training_early_stopping == 2:
      logging.info('global training early stopping')
      exit(0)
    
    if now_layers < args.max_layers:
      model.add_top_one_supercell()
      model.is_all_cell_normal_cell = False
      # model._layers already add 1 
    model = model.cuda()
    now_layers += 1
    if now_layers > args.max_layers:
      # stop adding cell, satisfy the args.max_layers restrict 
      logging.info('stop adding cell, due to the args.max_layers = %d restrict', args.max_layers)
      logging.info('final latency = %fms', model.all_normal_cell_latency)

  '''
  logging.info('start final training: %d epochs', args.final_training_epochs)
  #criterion = nn.CrossEntropyLoss()
  criterion = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )
  
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.final_training_epochs))

  train_x, train_label = load_epoch_data(flow_dict, 'train')
  test_x, test_label = load_epoch_data(flow_dict, 'test')

  train_data = Dataset(x=train_x, label=train_label)
  valid_data = Dataset(x=test_x, label=test_label)

  train_queue = torch.utils.data.DataLoader(
      train_data, collate_fn=paired_collate_fn, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, collate_fn=paired_collate_fn, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0)

  best_acc = 0.0
  #model.is_all_cell_normal_cell = True
  for epoch in range(args.final_training_epochs):
    logging.info('epoch %d lr %e', epoch, scheduler.get_last_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.final_training_epochs

    train_acc, _ = normal_train(train_queue, model, criterion, optimizer)
    logging.info('train_acc %f', train_acc)
    scheduler.step()

    valid_acc, _ = normal_infer(valid_queue, model, criterion)

    if valid_acc > best_acc:
      best_acc = valid_acc
    logging.info('valid_acc %f, best_acc %f', valid_acc, best_acc)

    #utils.save(model, os.path.join(args.save, 'weights.pt'))
  '''


def mid_training_for_eval(model, flow_dict, mid_eval_epochs, numlayer):
  #criterion = nn.CrossEntropyLoss()
  criterion = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(mid_eval_epochs))

  train_x, train_label = load_epoch_data(flow_dict, 'train')
  test_x, test_label = load_epoch_data(flow_dict, 'test')

  train_data = Dataset(x=train_x, label=train_label)
  test_data = Dataset(x=test_x, label=test_label)

  #train_queue = torch.utils.data.DataLoader(
  #    train_data, collate_fn=paired_collate_fn, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)

  test_queue = torch.utils.data.DataLoader(
      test_data, collate_fn=paired_collate_fn, batch_size=args.batch_size * 2, shuffle=False, pin_memory=True, num_workers=0)
  
  num_train = len(train_data)
  indices = list(np.random.choice(num_train, size=num_train, replace=False))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, collate_fn=paired_collate_fn,
      batch_size=args.batch_size * 2,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=0)
      # num_workers=2 will cause error and slow down the running

  valid_queue = torch.utils.data.DataLoader(
      train_data, collate_fn=paired_collate_fn,
      batch_size=args.batch_size * 2,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=0)

  valid_best_acc = 0.0
  test_best_acc = 0.0
  for epoch in range(mid_eval_epochs):
    logging.info('mid eval epoch %d lr %e', epoch, scheduler.get_last_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / mid_eval_epochs

    train_acc, _ = normal_train(train_queue, model, criterion, optimizer)
    logging.info('train_acc %f', train_acc)
    scheduler.step()

    valid_acc, _ = normal_infer(valid_queue, model, criterion)
    test_acc, _ = normal_infer(test_queue, model, criterion)

    if valid_acc > valid_best_acc:
      valid_best_acc = valid_acc
    if test_acc > test_best_acc:
      test_best_acc = test_acc
      utils.save(model, os.path.join(args.save, 'layer%d.pt' % (numlayer)))
      
    logging.info('valid_acc %f, valid_best_acc %f', valid_acc, valid_best_acc)
    logging.info('test_acc %f, test_best_acc %f', test_acc, test_best_acc)

    #utils.save(model, os.path.join(args.save, 'weights.pt'))
  
  return valid_best_acc


def train(epoch, train_queue, valid_queue, model, architect, criterion, optimizer, lr):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  latency = utils.AvgrageMeter()
  parameter = utils.AvgrageMeter()
  relu_latency = utils.AvgrageMeter()
  relu_parameter = utils.AvgrageMeter()
  total_loss = utils.AvgrageMeter()

  model.train()

  for step, (input, target) in enumerate(train_queue):
    n = input.size(0)

    input, target = input.cuda(), target.cuda()

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search, target_search = input_search.cuda(), target_search.cuda()
    
    architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    logits, expected_latency, expected_parameter = model(input)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    latency.update(expected_latency.item())
    parameter.update(expected_parameter.item())

    dec_rate = model.dec_rate
    ref_value_latency = model.init_latency * dec_rate
    ref_value_parameter = model.init_parameter * dec_rate
    reg_lambda = model.reg_lambda

    reg_loss_latency = reg_lambda * (expected_latency - ref_value_latency) / ref_value_latency
    reg_loss_parameter = reg_lambda * (expected_parameter - ref_value_parameter) / ref_value_parameter

    relu_latency.update(model.relu_func(reg_loss_latency).item())
    relu_parameter.update(model.relu_func(reg_loss_parameter).item())
    total_loss.update(loss.item() + model.relu_func(reg_loss_latency).item() + model.relu_func(reg_loss_parameter).item())

    if step % args.report_freq == 0:
      logging.info('train %03d  %f  %f  %f  %f  %f  %f  %f', step, objs.avg, top1.avg, latency.avg, parameter.avg, 
                                                             relu_latency.avg, relu_parameter.avg, total_loss.avg)
      
      #objs.reset()
      #top1.reset()
      latency.reset()
      parameter.reset()
      relu_latency.reset()
      relu_parameter.reset()
      total_loss.reset()
    
    #if step == 40:
      #break
    
    if step > 0 and step % 10 == 0 and args.use_loss_regulation == True:
      skip_num = 0
      for i in model.genotype().normal: # type model.genotype().normal is list, member is tuple 
        if i[0] == 'skip_connect' or 'pool' in i[0]:
        #if i[0] == 'skip_connect':
          skip_num += 1

      #if skip_num > 1.5:
      if skip_num > 0.5:
        logging.info('skip_or_pool_num = %d, early stopping', skip_num)
        break

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  latency = utils.AvgrageMeter()
  parameter = utils.AvgrageMeter()
  relu_latency = utils.AvgrageMeter()
  relu_parameter = utils.AvgrageMeter()
  total_loss = utils.AvgrageMeter()

  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    with torch.no_grad():
      input, target = input.cuda(), target.cuda()

      logits, expected_latency, expected_parameter = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      latency.update(expected_latency.item())
      parameter.update(expected_parameter.item())

      dec_rate = model.dec_rate
      ref_value_latency = model.init_latency * dec_rate
      ref_value_parameter = model.init_parameter * dec_rate
      reg_lambda = model.reg_lambda

      reg_loss_latency = reg_lambda * (expected_latency - ref_value_latency) / ref_value_latency
      reg_loss_parameter = reg_lambda * (expected_parameter - ref_value_parameter) / ref_value_parameter

      relu_latency.update(model.relu_func(reg_loss_latency).item())
      relu_parameter.update(model.relu_func(reg_loss_parameter).item())
      total_loss.update(loss.item() + model.relu_func(reg_loss_latency).item() + model.relu_func(reg_loss_parameter).item())

    if step % args.report_freq == 0:
      logging.info('valid %03d  %f  %f  %f  %f  %f  %f  %f', step, objs.avg, top1.avg, latency.avg, parameter.avg, 
                                                             relu_latency.avg, relu_parameter.avg, total_loss.avg)

      #objs.reset()
      #top1.reset()
      latency.reset()
      parameter.reset()
      relu_latency.reset()
      relu_parameter.reset()
      total_loss.reset()
    
    #if step == 40:
      #break

  return top1.avg, objs.avg


def normal_train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input, target = input.cuda(), target.cuda()
    #print(input.shape, target.shape)

    optimizer.zero_grad()

    logits, _, _ = model(input)

    loss = criterion(logits, target)
    
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, _ = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)

    if step % args.report_freq == 0:
      logging.info('train  step = %03d  loss = %f  acc = %f', step, objs.avg, top1.avg)

  return top1.avg, objs.avg


def normal_infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  infer_latency = utils.AvgrageMeter()
  final_true=[]
  final_pridict=[]
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    with torch.no_grad():
      input, target = input.cuda(), target.cuda()

      #torch.cuda.synchronize()
      start = time.time()

      logits, _, _ = model(input)

      #torch.cuda.synchronize()
      end = time.time()

      loss = criterion(logits, target)

      batch_len = target.shape[0]
      for j in range(0, batch_len):
          final_pridict.append(logits.argmax(1)[j].cpu())
          final_true.append(target[j].cpu())

      prec1, _ = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      infer_latency.update(end - start)

    if step % args.report_freq == 0:
      logging.info('valid  step = %03d  loss = %f  acc = %f  %fms', step, objs.avg, top1.avg, infer_latency.avg * 1e3)

      infer_latency.reset()
    
  logging.info(classification_report(final_true, final_pridict, digits = 4))
  logging.info(confusion_matrix(final_true, final_pridict))

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 