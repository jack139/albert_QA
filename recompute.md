## 使用重计算减少GPU内存占用

### 简述
​		使用GPU进行模型训练时会受到GPU内存的限制，内存太小则不能进行大模型的训练。在Bert族模型训练时，GPU内存太小会限制不能使用过大的batch size。比较大的batch size可以提高收敛速度和模型性能。



### 重计算原理
​		重计算方法在2016年被提出，最早出现在论文[Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)。这篇论文用时间换空间的思想，**在前向传播时只保存部分中间节点，在反向传播时重新计算没保存的部分**。论文通过这种机制，在每个batch只多计算一次前向的情况下，把n层网络的GPU内存占用优化到了O(sqrt(N))。在极端情况下，仍可用O(nlogn)的计算时间换取到O(logn)的GPU内存占用。论文中最重要的三点结论：**1. 梯度计算等价，理论上没有精度损失；** **2. 可以节省4倍+的GPU内存开销；** **3. 训练速度仅仅会被拖慢30%**。



### 实现方法
1. 参考开源实现

    https://github.com/cybertronai/gradient-checkpointing/

2. 替换梯度计算
  原开源实现通过替换```tf.gradients```实现重计算过程，因为我们使用```tf.train.Optimizer```进行梯度计算，而```tf.train.Optimizer```在内部条用了```tf.gradients```的一个别名```gradients.gradients```，因此做如下替换：
```python
import memory_saving_gradients
from tensorflow.python.ops import gradients
def gradients_memory(ys, xs, grad_ys=None, **kwargs):
    return memory_saving_gradients.gradients(ys, xs, grad_ys, checkpoints='collection', **kwargs)
gradients.__dict__["gradients"] = gradients_memory
```

3. 在模型中添加```checkpoints```用于标记重计算的位置
通过测试，只需要在模型中的```attention```层的输出位置进行重计算即可达到减少内存占用的目的。
```python
tf.add_to_collection('checkpoints', attention_output)
```



### 效果

​		在对Bert和Albert模型进行标记```checkpoints```后，对比测试结果如下（max_seq_len=512）：

| BERT 最大可用batch_size | RTX-2070S (8G) | Tesla T4 (16G) |
| ----------------------- | -------------- | -------------- |
| 未使用重计算            | 2              | 16             |
| 使用重计算              | 40             | 96             |

| ALBERT 最大可用batch_size | RTX-2070S (8G) | Tesla T4 (16G) |
| ------------------------- | -------------- | -------------- |
| 未使用重计算              | 16             | 32             |
| 使用重计算                | 72             | 136            |

