# coding=utf-8

import tensorflow as tf
import os
from models.albert_zh_modeling import AlbertModelMRC
from models.albert_zh_modeling import BertConfig as AlbertConfig
import utils
from tensorflow.python.framework import graph_util

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

max_seq_length = 512
bert_config = AlbertConfig.from_json_file('../nlp_model/albert_tiny_489k/albert_config_tiny.json')
input_ids = tf.placeholder(tf.int32, shape=[None, max_seq_length], name='input_ids')
segment_ids = tf.placeholder(tf.int32, shape=[None, max_seq_length], name='segment_ids')
input_mask = tf.placeholder(tf.float32, shape=[None, max_seq_length], name='input_mask')
eval_model = AlbertModelMRC(config=bert_config,
                          is_training=False,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          token_type_ids=segment_ids,
                          use_float16=False)

utils.init_from_checkpoint('../nlp_model/albert_tiny_489k/albert_model.ckpt')

config = tf.ConfigProto()
config.allow_soft_placement = True
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    with tf.gfile.FastGFile('model.pb', 'wb') as f:
        graph_def = sess.graph.as_graph_def()
        #output_nodes = ['start_logits', 'end_logits']
        output_nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
        print('outputs:', output_nodes)
        output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, output_nodes)
        f.write(output_graph_def.SerializeToString())