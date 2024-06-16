from collections import namedtuple

#Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
Genotype = namedtuple('Genotype', 'normal normal_concat')

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

DARTS = Genotype([('dil_conv_5x5', 0), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 3)], range(1, 5))

'''
DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])
search_epoch_99 = Genotype(normal=[('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('skip_connect', 1), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 1), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 2), ('max_pool_3x3', 0), ('skip_connect', 3), ('max_pool_3x3', 0), ('skip_connect', 3)], reduce_concat=range(2, 6))

search_EXP_20220310_080605 = Genotype(normal=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 2), ('max_pool_3x3', 0), ('sep_conv_3x3', 3), ('dil_conv_5x5', 2), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))
search_EXP_20220310_080841 = Genotype(normal=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('max_pool_3x3', 1), ('skip_connect', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 1), ('dil_conv_5x5', 0)], reduce_concat=range(2, 6))
search_EXP_20220310_102423 = Genotype(normal=[('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 3), ('dil_conv_5x5', 2), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('skip_connect', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 0)], reduce_concat=range(2, 6))

new_small_test = Genotype(normal=[('dil_conv_5x5', 0), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2)], 
                          normal_concat=range(1, 5), 
                          reduce=[('dil_conv_5x5', 0), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2)], 
                          reduce_concat=range(1, 5))

res = Genotype(normal=[('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 2)], normal_concat=range(1, 5), reduce=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 3)], reduce_concat=range(1, 5))

DARTS = res
'''