# coding=utf-8

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

corpus = [
        "《战国无双3》（）是由光荣和ω-force开发的战国无双系列的正统第三续作。本作以三大故事为主轴，\
分别是以武田信玄等人为主的《关东三国志》，织田信长等人为主的《战国三杰》，石田三成等人为主的《关原的年轻\
武者》，丰富游戏内的剧情。此部份专门介绍角色，欲知武器情报、奥义字或擅长攻击类型等，请至战国无双系列1.由\
于乡里大辅先生因故去世，不得不寻找其他声优接手。从猛将传 and Z开始。2.战国无双 编年史的原创男女主角亦有\
专属声优。此模式是任天堂游戏谜之村雨城改编的新增模式。本作中共有20张战场地图（不含村雨城），后来发行的猛\
将传再新增3张战场地图。但游戏内战役数量繁多，部分地图会有兼用的状况，战役虚实则是以光荣发行的2本「战国无\
双3 人物真书」内容为主，以下是相关介绍。（注：前方加☆者为猛将传新增关卡及地图。）合并本篇和猛将传的内容，\
村雨城模式剔除，战国史模式可直接游玩。主打两大模式「战史演武」&「争霸演武」。系列作品外传作品", 
        "广茂铁路是中国广东省一条起自广州市广州西站，向西跨越北江、西江，经佛山、三水、肇庆、云浮、阳江至\
茂名市茂名站的铁路，全长364.6公里，由三茂铁路股份有限公司管理运营。广茂铁路在广州由广茂联络线（前称流西联络\
线）连接广州西站及广州站，从而与京广、广深线连接，在茂名与黎湛铁路茂名支线连接，是珠江三角洲通往粤西南和雷州\
半岛的主要铁路干线。另有腰古至云浮，和三水站（原称西南站）至三水南站的支线，如将此两条支线计算在内，广茂铁路\
则全长421.326公里，共有47个车站。广茂铁路是在2004年2月29日起，因广州铁路集团控股的三茂铁路股份有限公司管\
辖范围增加原由广铁集团羊城铁路总公司管辖的广三铁路，原广三铁路和三茂铁路合称广茂铁路，原属广三线上的广州西站\
、石围塘站、佛山车务段和工务、电务、水电等部门由三茂铁路股份有限公司接管。同时中华人民共和国铁道部将广茂铁路\
由二级线路调整为一级线路。广三铁路于1903年筑成，全长49公里。而全长357公里的三茂铁路则于1991年投入运营。", 
        "大莱龙铁路位于山东省北部环渤海地区，西起位于益羊铁路的潍坊大家洼车站，向东经海化、寿光、寒亭、昌\
邑、平度、莱州、招远、终到龙口，连接山东半岛羊角沟、潍坊、莱州、龙口四个港口，全长175公里，工程建设概算总投\
资11.42亿元。铁路西与德大铁路、黄大铁路在大家洼站接轨，东与龙烟铁路相连。大莱龙铁路于1997年11月批复立项，\
2002年12月28日全线铺通，2005年6月建成试运营，是横贯山东省北部的铁路干线德龙烟铁路的重要组成部分，构成山东\
省北部沿海通道，并成为环渤海铁路网的南部干线。铁路沿线设有大家洼站、寒亭站、昌邑北站、海天站、平度北站、沙河\
站、莱州站、朱桥站、招远站、龙口西站、龙口北站、龙口港站。大莱龙铁路官方网站", 
        "易惠科技基于易联众集团的业务基础与技术沉淀，持续拓展和完善服务网络，目前在北京市设有研发中心和分支\
机构，在福州市、安徽省和山西省设有3家分公司。公司团队成员在医疗健康信息化方面拥有多年的行业积淀，具备国内领先\
的项目实施、运营管理服务经验和能力。目前，公司在医疗健康信息化领域已服务了16个省份，300+个医疗机构，承建的项\
目成功帮助客户多次获评国家级、省级优秀案例和创新项目奖项。我们将秉承“专业、创新、满意”的服务理念持续发力，为客\
户量身打造可持续的解决方案和全方位的运营服务。"\
]

questions = [
    "易惠科技的服务理念是什么？",
    "易惠科技有几家分公司？",
    "易惠科技有哪几家分公司？",
    "广茂铁路有多长？"
]

# 装入Albert模型标签
tokenizer = tokenization.BertTokenizer(vocab_file='../nlp_model/albert_zh_base/vocab_chinese.txt',
                                       do_lower_case=True)

# 初始化文档获取类
corpus_rank = CORPUS_RANK(corpus)

with tf.Session(graph=p_graph) as sess:

    # 开始循环答题
    for question in questions:

        # BM25 获取相关文本
        max_index = corpus_rank.get_document(question)
        print(max_index)
        context = corpus[max_index]

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

        start_logits_, end_logits_ = sess.run([start_logits, end_logits], feed_dict={input_ids: input_ids_,
                                                                                     segment_ids: segment_ids_,
                                                                                     input_mask: input_mask_})
        st = np.argmax(start_logits_[0, :])
        ed = np.argmax(end_logits_[0, :])
        print('Question: ', question)
        print('Answer:', "".join(input_tokens[st:ed + 1]))
