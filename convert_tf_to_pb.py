# coding=utf-8

import tensorflow as tf
import os
#from models.albert_google_modeling import AlbertModelMRC, AlbertConfig
from models.albert_zh_modeling import AlbertModelMRC, BertConfig as AlbertConfig
import utils
from tensorflow.python.framework import graph_util

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

max_seq_length = 512
#bert_config = AlbertConfig.from_json_file('../nlp_model/albert_zh_base/albert_config.json')
bert_config = AlbertConfig.from_json_file('../nlp_model/albert_large_zh/albert_config_large.json')
input_ids = tf.placeholder(tf.int32, shape=[None, max_seq_length], name='input_ids')
segment_ids = tf.placeholder(tf.int32, shape=[None, max_seq_length], name='segment_ids')
input_mask = tf.placeholder(tf.float32, shape=[None, max_seq_length], name='input_mask')
eval_model = AlbertModelMRC(config=bert_config,
                          is_training=False,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          token_type_ids=segment_ids,
                          use_float16=False)

utils.init_from_checkpoint(
  #'outputs/cmrc2018/albert_zh_base/epoch2_batch16_lr3e-05_warmup0.1_anslen50_tf/checkpoint_score_F1-86.195_EM-65.02.ckpt')
  'outputs/cmrc2018/albert_large_zh/epoch2_batch3_lr3e-05_warmup0.1_anslen50_tf/checkpoint_score_F1-86.754_EM-66.511.ckpt')

config = tf.ConfigProto()
config.allow_soft_placement = True
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    # 输出 pb
    with tf.gfile.FastGFile('model.pb', 'wb') as f:
        graph_def = sess.graph.as_graph_def()
        output_nodes = ['finetune_mrc/Squeeze', 'finetune_mrc/Squeeze_1']
        print('outputs:', output_nodes)
        #print('\n'.join([n.name for n in tf.get_default_graph().as_graph_def().node])) # 所有层的名字
        output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, output_nodes)
        f.write(output_graph_def.SerializeToString())

    # save_model 输出 , for goland 测试
    builder = tf.saved_model.builder.SavedModelBuilder("outputs/saved-model")
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.TRAINING], clear_devices=True)
    builder.save()  
