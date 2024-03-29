# coding=utf-8
import re
from datetime import datetime
import numpy as np
import tensorflow as tf
import tokenizations.official_tokenization as tokenization
from preprocess.bm25 import CORPUS_RANK

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

corpus = [ # 至少要有3条
    "易惠科技基于易联众集团的业务基础与技术沉淀，持续拓展和完善服务网络，目前在北京市设有研发中心和分支\
机构，在福州市、安徽省和山西省设有3家分公司。公司团队成员在医疗健康信息化方面拥有多年的行业积淀，具备国内领先\
的项目实施、运营管理服务经验和能力。目前，公司在医疗健康信息化领域已服务了16个省份，300+个医疗机构，承建的项\
目成功帮助客户多次获评国家级、省级优秀案例和创新项目奖项。我们将秉承“专业、创新、满意”的服务理念持续发力，为客\
户量身打造可持续的解决方案和全方位的运营服务。",
    "深度学习（英语：deep learning）是机器学习的分支，是一种以人工神经网络为架构，对资料进行表征学习\
的算法。深度学习是机器学习中一种基于对数据进行表征学习的算法。观测值（例如一幅图像）可以使用多种方式来表示，如\
每个像素强度值的向量，或者更抽象地表示成一系列边、特定形状的区域等。而使用某些特定的表示方法更容易从实例中学习\
任务（例如，人脸识别或面部表情识别）。深度学习的好处是用非监督式或半监督式的特征学习和分层特征提取高效算法\
来替代手工获取特征。",
    "深度学习的基础是机器学习中的分散表示（distributed representation）。分散表示假定观测值是由不同因\
子相互作用生成。在此基础上，深度学习进一步假定这一相互作用的过程可分为多个层次，代表对观测值的多层抽象。不同\
的层数和层的规模可用于不同程度的抽象。深度学习运用了这分层次抽象的思想，更高层次的概念从低层次的概念学习得到。\
这一分层结构常常使用贪心算法逐层构建而成，并从中选取有助于机器学习的更有效的特征。不少深度学习算法都以无监督\
学习的形式出现，因而这些算法能被应用于其他算法无法企及的无标签数据，这一类数据比有标签数据更丰富，也更容易获\
得。这一点也为深度学习赢得了重要的优势。",
    "胡夫金字塔（阿拉伯语：هرم أكبر‎，希腊语：Πυραμίδες της Γκίζα，英文：Pyramid of Khufu）又称吉\
萨大金字塔，是位于埃及吉萨三座著名的金字塔中最为古老也是最大的一座。同时也是古代世界七大奇跡唯一尚存的建筑物",
]

questions = [
    "易惠科技的服务理念是什么？",
    "易惠科技有几家分公司？",
    "易惠科技有哪几家分公司？",
    "什么是深度学习？",
    "深度学习的优点是什么？",
    "胡夫金字塔你是谁？",
    "你是谁？"
]

# 清除文本中非中文非英文的字符（例如：日文、阿拉伯文） --- 中文模型的albert里没有这些token, 会导致go调用模型时内存溢出
def cleantxt(raw):
    fil = re.compile(u'[^0-9a-zA-Z\u4e00-\u9fa5.，,。？?“”"()（）$￥%!！:：、/|]+', re.UNICODE)
    return fil.sub(' ', raw) 

# 装入Albert模型标签
tokenizer = tokenization.BertTokenizer(vocab_file='../nlp_model/albert_zh_base/vocab_chinese.txt',
                                       do_lower_case=True)

# 初始化文档获取类
corpus_rank = CORPUS_RANK(corpus)

with tf.Session(graph=p_graph) as sess:

    # 开始循环答题
    for question in questions:

        print('\n################## ')

        question = cleantxt(question)

        # BM25 获取相关文本
        max_index = corpus_rank.get_document(question)

        print(max_index)

        for index, value in max_index:
            if value==0: # bm25 score是0
                print('Question: ', question, '\n ---> 未找到答案')
                break

            print(index)
            context = cleantxt(corpus[index])

            context = context.replace('”', '"').replace('“', '"')
            question = question.replace('”', '"').replace('“', '"')

            # 进行问答预测
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

            input_ids_ = np.array(input_ids_).reshape(1, max_seq_length)
            segment_ids_ = np.array(segment_ids_).reshape(1, max_seq_length)
            input_mask_ = np.array(input_mask_).reshape(1, max_seq_length)

            #print(input_ids_)
            #print(segment_ids_)
            #print(input_mask_)

            start_time = datetime.now()
            start_logits_, end_logits_ = sess.run([start_logits, end_logits], feed_dict={input_ids: input_ids_,
                                                                                         segment_ids: segment_ids_,
                                                                                         input_mask: input_mask_})
            st = np.argmax(start_logits_[0, :])
            ed = np.argmax(end_logits_[0, :])
            print('[Time taken: {!s}]'.format(datetime.now() - start_time))
            print(st, ed)


            # 判断一个unicode是否是英文字母
            def is_alphabet(uchar):
                if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
                    return True
                else:
                    return False
            # 处理token中的英文，例如： 'di', '##st', '##ri', '##bu', '##ted', 're', '##pr', '##ese', '##nt', '##ation',
            ans = "".join([i[2:] if i.startswith('##') else (' '+i if is_alphabet(i[0]) else i)  for i in input_tokens[st:ed + 1]])

            # 不处理：
            #ans = "".join(input_tokens[st:ed + 1])  

            if not ans.startswith('[CLS]'): # 找到答案
                print('Question: ', question)
                print('Answer:', ans)
                break
