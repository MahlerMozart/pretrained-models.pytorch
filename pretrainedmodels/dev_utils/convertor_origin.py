from collections import OrderedDict
import numpy as np
#from nn_transfer import transfer, util
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from pretrainedmodels.models import nasnet_mobile as nasnet
import numpy as np
import h5py
import keras
from keras.models import load_model
from keras.utils import plot_model
from keras.applications.nasnet import NASNetMobile
from tensorboardX import SummaryWriter
import torchvision
import json


py_model = nasnet.nasnetamobile(num_classes=1000, pretrained=False)
state_dict = py_model.state_dict()
keys_names = [x for x in state_dict.keys()]

# z = (c)
# writer.add_graph(py_model, z)
# torch.onnx.export(py_model, z, "test.proto", verbose=True)
# writer.add_graph_onnx("test.proto")
 # writer.close()


model = NASNetMobile()
model.load_weights('./NASNet-mobile_from_keras_git.h5')

# print(model.summary())
model_weights = model.weights
names = [weight.name for layer in model.layers for weight in layer.weights]
keras_weights = model.get_weights()
keras_dict = OrderedDict((names[i], keras_weights[i]) for i in range(len(names)))


main_block_dict = {
    'stem_conv1'                 : ['conv0', 1, 'reduction'],
    'stem_bn1'                   : ['conv0', 2, 'reduction'],
    'reduction_A_block_stem_1'   : ['cell_stem_0', 1, 'reduction'],
    'reduction_A_block_stem_2'   : ['cell_stem_1', 2, 'reduction'],
    'normal_A_block_0'           : ['cell_0', 2, 'normal'],
    'normal_A_block_1'           : ['cell_1', 3, 'normal'],
    'normal_A_block_2'           : ['cell_2', 3, 'normal'],
    'normal_A_block_3'           : ['cell_3', 3, 'normal'],
    'reduction_A_block_reduce_4' : ['reduction_cell_0', 3, 'reduction'],
    'normal_A_block_5'           : ['cell_6', 2, 'normal'],
    'normal_A_block_6'           : ['cell_7', 3, 'normal'],
    'normal_A_block_7'           : ['cell_8', 3, 'normal'],
    'normal_A_block_8'           : ['cell_9', 3, 'normal'],
    'reduction_A_block_reduce_8' : ['reduction_cell_1', 3, 'reduction'],
    'normal_A_block_9'           : ['cell_12', 2, 'normal'],
    'normal_A_block_10'          : ['cell_13', 3, 'normal'],
    'normal_A_block_11'          : ['cell_14', 3, 'normal'],
    'normal_A_block_12'          : ['cell_15', 3, 'normal']
}


second_adjust = {
        'path_1.avgpool': 'adjust_avg_pool_1_{}',
        'path_1.conv'   : 'adjust_conv_1_{}',
        'path_2.avgpool': 'adjust_avg_pool_2_{}',
        'path_2.conv'   : 'adjust_conv_2_{}',
        'final_path_bn' : 'adjust_bn_{}',
        'conv_prev_1x1.conv': "adjust_conv_projection_{}",
        'conv_prev_1x1.bn'  : 'adjust_bn_{}'}


reduction_main = {
    'conv_1x1.conv'     : ["reduction_A_block", "separable_conv_block_reduction_conv_1_{}", "reduction_conv"],
    'conv_1x1.bn'       : ["reduction_A_block", "separable_conv_block_reduction_bn_1_{}", 'reduction_bn'],
    'comb_iter_0_left'   : ["block_1", "separable_conv_block_reduction_left1_{}", ''],
    'comb_iter_0_right'  : ["block_1", "separable_conv_block_reduction_1_{}", ''],
    'comb_iter_1_right'  : ["block_2", "separable_conv_block_reduction_right2_{}", ''],
    'comb_iter_2_right'  : ["block_3", "separable_conv_block_reduction_right3_{}", ''],
    'comb_iter_4_left'   : ["block_5", "separable_conv_block_reduction_left4_{}", ''],
}


normal_main = {
    'conv_1x1.conv'     : ['normal_A_block', 'separable_conv_block_normal_conv_1_{}', "normal_conv"],
    'conv_1x1.bn'       : ['normal_A_block', 'separable_conv_block_normal_bn_1_{}', 'normal_bn'],
    'comb_iter_0_left'  : ['block_1', 'separable_conv_block_normal_left1_{}', ''],
    'comb_iter_0_right' : ['block_1', 'separable_conv_block_normal_right1_{}', ''],
    'comb_iter_1_left'  : ['block_2', 'separable_conv_block_normal_left2_{}', ''],
    'comb_iter_1_right' : ['block_2', 'separable_conv_block_normal_right2_{}', ''],
    'comb_iter_4_left'  : ['block_5', 'separable_conv_block_normal_left5_{}', '']
}


def find_value(value, ll):
    output = [x.endswith(value) for x in ll]
    if True in output:
        return True
    else:
        return False


def create_dict_pytorch(keras_long_name):
    keras_names_list = keras_long_name.split(
        "/")  # ['reduction_A_block_stem_1', 'block_1', 'separable_conv_block_reduction_1_stem_1', 'separable_conv_1_reduction_1_stem_1', 'depthwise_kernel:0']
    pytorch_block_info = main_block_dict[keras_names_list[0]]  # ['cell_stem_0', 1, 'reduction']
    pytorch_long_dict = {}
    pytorch_long_dict['first_name'] = pytorch_block_info[0]  # 'cell_stem_0'
    reduction_blocks_names = [y[0] for x, y in reduction_main.items()]
    normal_blocks_names = [y[0] for x, y in normal_main.items()]
    if (pytorch_block_info[2] == 'reduction') and (keras_names_list[1] in reduction_blocks_names):

        if keras_names_list[0].split("_")[-2] in ['stem', 'reduce']:
            block_name = "_".join([keras_names_list[0].split("_")[-2],
                                           keras_names_list[0].split("_")[-1]])
        else:
            block_name = keras_names_list[0].split("_")[-1]

        lKey = [key for key, value in reduction_main.items()
                    if "_".join([value[1][:-3], block_name]) == keras_names_list[2]]
        pytorch_long_dict['second_name'] = lKey[0]
        pytorch_long_dict['tail_name'] = keras_names_list[-1].split(":")[0]
        pytorch_long_dict['index'] = keras_names_list[3].split("_")[2]
    elif (pytorch_block_info[2] == 'normal') and (keras_names_list[1] in normal_blocks_names):
        if keras_names_list[0].split("_")[-2] in ['stem', 'reduce']:
            block_name = "_".join([keras_names_list[0].split("_")[-2],
                                           keras_names_list[0].split("_")[-1]])
        else:
            block_name = keras_names_list[0].split("_")[-1]
        lKey = [key for key, value in normal_main.items()
                if "_".join([value[1][:-3], block_name]) == keras_names_list[2]]
        pytorch_long_dict['second_name'] = lKey[0]
        pytorch_long_dict['tail_name'] = keras_names_list[-1].split(":")[0]
        pytorch_long_dict['index'] = keras_names_list[3].split("_")[2]

    elif keras_names_list[1] == 'adjust_block':
        if keras_names_list[0].split("_")[-2] in ['stem', 'reduce']:
            block_name = "_".join([keras_names_list[0].split("_")[-2],
                                           keras_names_list[0].split("_")[-1]])
        else:
            block_name = keras_names_list[0].split("_")[-1]
        lKey = [key for key, value in second_adjust.items()
                        if "_".join([value[:-3], block_name]) == keras_names_list[3]]
        if lKey == ['final_path_bn', 'conv_prev_1x1.bn']:
            if pytorch_block_info[1] == 2:
                lKey = 'final_path_bn'
            elif pytorch_block_info[1] == 3:
                lKey = 'conv_prev_1x1.bn'

        if not isinstance(lKey, list):
            pytorch_long_dict['second_name'] = lKey
        else:
            pytorch_long_dict['second_name'] = lKey[0]
        pytorch_long_dict['tail_name'] = keras_names_list[-1].split(":")[0]
        pytorch_long_dict['index'] = keras_names_list[3].split("_")[2]

    elif "A_block" in keras_names_list[0]:
        if keras_names_list[0].split("_")[0] == 'reduction':
            kkey = "_".join([keras_names_list[1].split("_")[0], keras_names_list[1].split("_")[1]])
            lKey = [key for key, value in reduction_main.items()
                        if value[2] == kkey]
        elif keras_names_list[0].split("_")[0] == 'normal':
            kkey = "_".join([keras_names_list[1].split("_")[0], keras_names_list[1].split("_")[1]])
            lKey = [key for key, value in normal_main.items()
                    if value[2] == kkey]

        pytorch_long_dict['second_name'] = lKey[0]
        pytorch_long_dict['tail_name'] = keras_names_list[-1].split(":")[0]
        pytorch_long_dict['index'] = None

    return pytorch_long_dict

print()


def dict_to_pytorch_name(l):
    if l['tail_name'] == 'depthwise_kernel':
        element = ".".join(['separable_{}'.format(l['index']), 'depthwise_conv2d', "weight"])
        pytorch_name = ".".join([l['first_name'], l['second_name'], element])
        if not pytorch_name in list(state_dict.keys()):
            print(pytorch_name)

    elif l['tail_name'] == 'pointwise_kernel':
        element = ".".join(['separable_{}'.format(l['index']), 'pointwise_conv2d', "weight"])
        pytorch_name = ".".join([l['first_name'], l['second_name'], element])
        if not pytorch_name in list(state_dict.keys()):
            print(pytorch_name)

    elif l['tail_name'] == 'moving_mean':
        if l['second_name'] in ['final_path_bn', 'conv_1x1.bn', 'conv_prev_1x1.bn']:
            element = 'running_mean'
        else:
            element = ".".join(['bn_sep_{}'.format(l['index']), 'running_mean'])
        pytorch_name = ".".join([l['first_name'], l['second_name'], element])
        if not pytorch_name in list(state_dict.keys()):
            print(pytorch_name)

    elif l['tail_name'] == 'moving_variance':
        if l['second_name'] in ['final_path_bn', 'conv_1x1.bn', 'conv_prev_1x1.bn']:
            element = 'running_var'
        else:
            element = ".".join(['bn_sep_{}'.format(l['index']), 'running_var'])
        pytorch_name = ".".join([l['first_name'], l['second_name'], element])
        if not pytorch_name in list(state_dict.keys()):
            print(pytorch_name)

    elif l['tail_name'] == 'gamma':
        if l['second_name'] in ['final_path_bn', 'conv_1x1.bn', 'conv_prev_1x1.bn']:
            element = 'weight'
        else:
            element = ".".join(['bn_sep_{}'.format(l['index']), 'weight'])
        pytorch_name = ".".join([l['first_name'], l['second_name'], element])
        if not pytorch_name in list(state_dict.keys()):
            print(pytorch_name)

    elif l['tail_name'] == 'beta':
        if l['second_name'] in ['final_path_bn', 'conv_1x1.bn', 'conv_prev_1x1.bn']:
            element = 'bias'
        else:
            element = ".".join(['bn_sep_{}'.format(l['index']), 'bias'])
        pytorch_name = ".".join([l['first_name'], l['second_name'], element])
        if not pytorch_name in list(state_dict.keys()):
            print(pytorch_name)

    elif l['tail_name'] == 'kernel':
        element = 'weight'
        pytorch_name = ".".join([l['first_name'], l['second_name'], element])
        if not pytorch_name in list(state_dict.keys()):
            print(pytorch_name)

    return pytorch_name


matcher_dict = {}
matcher_dict['stem_conv1/kernel:0'] = ['conv0.conv.weight',
                                       keras_dict['stem_conv1/kernel:0'].shape, state_dict['conv0.conv.weight'].shape]
matcher_dict['stem_bn1/gamma:0'] = ['conv0.bn.weight',
                                    keras_dict['stem_bn1/gamma:0'].shape, state_dict['conv0.bn.weight'].shape]
matcher_dict['stem_bn1/beta:0'] = ['conv0.bn.bias',
                                   keras_dict['stem_bn1/beta:0'].shape, state_dict['conv0.bn.bias'].shape]
matcher_dict['stem_bn1/moving_mean:0'] = ['conv0.bn.running_mean',
                                          keras_dict['stem_bn1/moving_mean:0'].shape, state_dict['conv0.bn.running_mean'].shape]
matcher_dict['stem_bn1/moving_variance:0'] = ['conv0.bn.running_var',
                                              keras_dict['stem_bn1/moving_variance:0'].shape, state_dict['conv0.bn.running_var'].shape]


for ind in range(len(list(keras_dict.keys()))):
    try:
        list_keys = list(keras_dict.keys())
        v = create_dict_pytorch(list_keys[ind])
        name_in_pytorch = dict_to_pytorch_name(v)
        matcher_dict[list_keys[ind]] = [name_in_pytorch, keras_dict[list_keys[ind]].shape, state_dict[name_in_pytorch].shape]
    except:
        print("Failed {} {}".format(ind, list(keras_dict.keys())[ind]))

matcher_dict['predictions/kernel:0'] = ['last_linear.weight', keras_dict['predictions/kernel:0'].shape, state_dict['last_linear.weight'].shape]
matcher_dict['predictions/bias:0'] = ['last_linear.bias', keras_dict['predictions/bias:0'].shape, state_dict['last_linear.bias'].shape]


pytorch_model = state_dict.copy()
for py_name in list(state_dict.keys()):
    keras_name = [key for key, value in matcher_dict.items() if value[0] == py_name]
    keras_shape = [value[1] for key, value in matcher_dict.items() if value[0] == py_name]
    pyt_shape = [value[2] for key, value in matcher_dict.items() if value[0] == py_name]
    try:
        weights = keras_dict[keras_name[0]]
        if len(weights.shape) == 4:
            if (py_name in ['conv0.conv.weight', 'cell_stem_0.conv_1x1.conv.weight', 'cell_stem_1.conv_1x1.conv.weight']) \
                    or ('pointwise' in py_name) or ("path_1" in py_name) or ("path_2" in py_name) or \
                    ("conv_prev_1x1" in py_name) or ("conv_1x1" in py_name):
                transformed_weights = weights.transpose(3, 2, 0, 1)
                assert pytorch_model[py_name].shape == transformed_weights.shape
                pytorch_model[py_name] = torch.FloatTensor(transformed_weights)
            else:
                transformed_weights = weights.transpose(2, 3, 0, 1)
                assert pytorch_model[py_name].shape == transformed_weights.shape
                pytorch_model[py_name] = torch.FloatTensor(transformed_weights)
        elif len(weights.shape) == 2:
            transformed_weights = weights.transpose(1, 0)
            assert pytorch_model[py_name].shape == transformed_weights.shape
            pytorch_model[py_name] = torch.FloatTensor(transformed_weights)
        else:
            transformed_weights = weights
            pytorch_model[py_name] = torch.FloatTensor(transformed_weights)
            assert pytorch_model[py_name].shape == transformed_weights.shape
    except IndexError:
        print(py_name)
    except:
        print("Failed_to_convert {} and {}".format(keras_name[0], py_name))
    pytorch_model[py_model] = transformed_weights


pytorch_model.pop(list(pytorch_model.keys())[-1])

for i in range(len(state_dict)):
    if not list(state_dict.keys())[i] == list(pytorch_model.keys())[i]:
        print("Error with {}".format(i))

for i in range(len(state_dict)):
    if not state_dict[list(state_dict.keys())[i]].shape == pytorch_model[list(pytorch_model.keys())[i]].shape:
        print("Error with {}".format(i))

assert len(state_dict) == len(pytorch_model)
torch.save(pytorch_model, "NASNet-mobile-keras-pytorch.pth.tar")

print()