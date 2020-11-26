# coding=utf-8

import tensorflow as tf
import os, shutil
import utils
from tensorflow.python.framework import graph_util

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# ------- ablert_google base
from models.albert_google_modeling import AlbertModelMRC, AlbertConfig
config_json_path = '../nlp_model/albert_zh_base/albert_config.json'
checkpoint_path = 'outputs/cmrc2018/albert_zh_base/epoch2_batch16_lr3e-05_warmup0.1_anslen50_tf/checkpoint_score_F1-86.195_EM-65.02.ckpt'

# ------- albert_zh large
#from models.albert_zh_modeling import AlbertModelMRC, BertConfig as AlbertConfig
#config_json_path = '../nlp_model/albert_large_zh/albert_config_large.json'
#checkpoint_path = 'outputs/cmrc2018/albert_large_zh/epoch2_batch6_lr3e-05_warmup0.1_anslen50_tf/checkpoint_score_F1-87.812_EM-67.071.ckpt'

# ------- albert_zh base
#from models.albert_zh_modeling import AlbertModelMRC, BertConfig as AlbertConfig
#config_json_path = '../nlp_model/albert_base_zh_36k/albert_config_base.json'
#checkpoint_path = 'outputs/cmrc2018/albert_base_zh_36k/epoch2_batch20_lr3e-05_warmup0.1_anslen50_tf/checkpoint_score_F1-85.748_EM-64.306.ckpt'

# ------- bert_google base
#from models.bert_google_modeling import BertModelMRC as AlbertModelMRC, BertConfig as AlbertConfig
#config_json_path = '../nlp_model/chinese_bert_L-12_H-768_A-12/bert_config.json'
#checkpoint_path = 'outputs/cmrc2018/chinese_L-12_H-768_A-12/epoch2_batch12_lr3e-05_warmup0.1_anslen50_tf/checkpoint_score_F1-85.51_EM-64.306.ckpt'

max_seq_length = 512
bert_config = AlbertConfig.from_json_file(config_json_path)
input_ids = tf.placeholder(tf.int32, shape=[None, max_seq_length], name='input_ids')
segment_ids = tf.placeholder(tf.int32, shape=[None, max_seq_length], name='segment_ids')
input_mask = tf.placeholder(tf.float32, shape=[None, max_seq_length], name='input_mask')
eval_model = AlbertModelMRC(config=bert_config,
                          is_training=False,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          token_type_ids=segment_ids,
                          use_float16=False)

utils.init_from_checkpoint(checkpoint_path)

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
    if os.path.exists('outputs/saved-model'):
        shutil.rmtree("outputs/saved-model") 
    builder = tf.saved_model.builder.SavedModelBuilder("outputs/saved-model")
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.TRAINING], clear_devices=True)
    builder.save()  
