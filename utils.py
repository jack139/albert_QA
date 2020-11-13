# coding=utf-8

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import collections
import re


def check_args(args, rank=0):
    args.setting_file = os.path.join(args.checkpoint_dir, args.setting_file)
    args.log_file = os.path.join(args.checkpoint_dir, args.log_file)
    if rank == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        with open(args.setting_file, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            print('------------ Options -------------')
            for k in args.__dict__:
                v = args.__dict__[k]
                opt_file.write('%s: %s\n' % (str(k), str(v)))
                print('%s: %s' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
            print('------------ End -------------')

    return args


def show_all_variables(rank=0):
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True if rank == 0 else False)


def get_assigment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    initialized_variable_names = {}
    new_variable_names = set()
    unused_variable_names = set()

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            if 'adam' not in name and 'lamb' not in name and 'accum' not in name:
                unused_variable_names.add(name)
            continue
        # assignment_map[name] = name
        assignment_map[name] = name_to_variable[name]
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    for name in name_to_variable:
        if name not in initialized_variable_names:
            new_variable_names.add(name)
    return assignment_map, initialized_variable_names, new_variable_names, unused_variable_names


# loading weights
def init_from_checkpoint(init_checkpoint, tvars=None, rank=0):
    if not tvars:
        tvars = tf.trainable_variables()
    assignment_map, initialized_variable_names, new_variable_names, unused_variable_names \
        = get_assigment_map_from_checkpoint(tvars, init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    if rank == 0:
        # 显示成功加载的权重
        for t in initialized_variable_names:
            if ":0" not in t:
                print("Loading weights success: " + t)

        # 显示新的参数
        print('New parameters:', new_variable_names)

        # 显示初始化参数中没用到的参数
        print('Unused parameters', unused_variable_names)
