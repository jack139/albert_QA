# coding=utf-8

import tensorflow as tf

with tf.gfile.FastGFile('model.pb', 'rb') as f:
    intput_graph_def = tf.GraphDef()
    intput_graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as p_graph:
        tf.import_graph_def(intput_graph_def)
        # 去掉模型中GPU信息，用于使用模型在CPU上预测
        g = tf.get_default_graph() 
        ops = g.get_operations() 
        for op in ops: 
            op._set_device('/device:CPU:*')

input_ids = p_graph.get_tensor_by_name("import/input_ids:0")
input_mask = p_graph.get_tensor_by_name('import/input_mask:0')
segment_ids = p_graph.get_tensor_by_name('import/segment_ids:0')
start_logits = p_graph.get_tensor_by_name('import/finetune_mrc/Squeeze:0')
end_logits = p_graph.get_tensor_by_name('import/finetune_mrc/Squeeze_1:0')

max_seq_length = 512

context = "易惠科技基于易联众集团的业务基础与技术沉淀，持续拓展和完善服 \
务网络，目前在北京市设有研发中心和分支机构，在福州市、安徽省和山 \
西省设有3家分公司。公司团队成员在医疗健康信息化方面拥有多年的 \
行业积淀，具备国内领先的项目实施、运营管理服务经验和能力。 \
目前，公司在医疗健康信息化领域已服务了16个省份，300+个医疗机 \
构，承建的项目成功帮助客户多次获评国家级、省级优秀案例和创新项目 \
奖项。我们将秉承“专业、创新、满意”的服务理念持续发力，为客户量 \
身打造可持续的解决方案和全方位的运营服务。"
context = context.replace('”', '"').replace('“', '"')

print("Context: ", context)

questions = [
    "易惠科技的服务理念是什么？",
    "易惠科技有几家分公司？",
    "易惠科技有哪几家分公司？"
]
questions = [i.replace('”', '"').replace('“', '"') for i in questions]

import tokenizations.official_tokenization as tokenization

tokenizer = tokenization.BertTokenizer(vocab_file='../nlp_model/albert_zh_base/vocab_chinese.txt',
                                       do_lower_case=True)

for question in questions:
    question_tokens = tokenizer.tokenize(question)
    context_tokens = tokenizer.tokenize(context)
    input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + context_tokens + ['[SEP]']
    print(len(input_tokens))
    input_ids_ = tokenizer.convert_tokens_to_ids(input_tokens)
    segment_ids_ = [0] * (2 + len(question_tokens)) + [1] * (1 + len(context_tokens))
    input_mask_ = [1] * len(input_tokens)

    while len(input_ids_) < max_seq_length:
        input_ids_.append(0)
        segment_ids_.append(0)
        input_mask_.append(0)

    import numpy as np

    input_ids_ = np.array(input_ids_).reshape(1, max_seq_length)
    segment_ids_ = np.array(segment_ids_).reshape(1, max_seq_length)
    input_mask_ = np.array(input_mask_).reshape(1, max_seq_length)

    with tf.Session(graph=p_graph) as sess:
        start_logits_, end_logits_ = sess.run([start_logits, end_logits], feed_dict={input_ids: input_ids_,
                                                                                     segment_ids: segment_ids_,
                                                                                     input_mask: input_mask_})
        st = np.argmax(start_logits_[0, :])
        ed = np.argmax(end_logits_[0, :])
        print('Question: ', question)
        print('Answer:', "".join(input_tokens[st:ed + 1]))
