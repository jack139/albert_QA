# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions and classes related to optimization (weight updates)."""

import os
# 启用AMP, Volta以下显卡，需要设置环境变量 TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE=1
# 原因来自：https://devtalk.nvidia.com/default/topic/1052688/container-tensorflow/
#               issue-about-no-suitable-gpus-detected-when-using-mixed-precision-graph-optimizer/
os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'

import re
import tensorflow as tf


class Optimizer(object):
    def __init__(self, loss, init_lr, num_train_steps, num_warmup_steps,
                 hvd=None, use_fp16=False, loss_count=1000, clip_norm=1.0,
                 beta1=0.9, beta2=0.999):
        """Creates an optimizer training op."""
        self.global_step = tf.train.get_or_create_global_step()

        # avoid step change in learning rate at end of warmup phase
        decayed_learning_rate_at_crossover_point = init_lr * (1.0 - float(num_warmup_steps) / float(num_train_steps))
        adjusted_init_lr = init_lr * (init_lr / decayed_learning_rate_at_crossover_point)
        learning_rate = tf.constant(value=adjusted_init_lr, shape=[], dtype=tf.float32)

        # Implements linear decay of the learning rate.
        learning_rate = tf.train.polynomial_decay(
            learning_rate,
            self.global_step,
            num_train_steps,
            end_learning_rate=0.0,
            power=1.0,
            cycle=False)

        # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
        # learning rate will be `global_step/num_warmup_steps * init_lr`.
        if num_warmup_steps:
            global_steps_int = tf.cast(self.global_step, tf.int32)
            warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

            global_steps_float = tf.cast(global_steps_int, tf.float32)
            warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

            warmup_percent_done = global_steps_float / warmup_steps_float
            warmup_learning_rate = init_lr * warmup_percent_done

            is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
            learning_rate = ((1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

        self.learning_rate = learning_rate

        # It is recommended that you use this optimizer for fine tuning, since this
        # is how the model was trained (note that the Adam m/v variables are NOT
        # loaded from init_checkpoint.)
        optimizer = AdamWeightDecayOptimizer(
            learning_rate=learning_rate,
            weight_decay_rate=0.01,
            beta_1=beta1,
            beta_2=beta2,
            epsilon=1e-6,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

        if use_fp16:
            # 启用AMP，需要有tensor core的Nvidia显卡
            # https://developer.nvidia.com/automatic-mixed-precision
            optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)


        tvars = tf.trainable_variables()
        grads_and_vars = optimizer.compute_gradients(loss, tvars)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        grads, tvars = list(zip(*grads_and_vars))
        all_are_finite = tf.reduce_all(
            [tf.reduce_all(tf.is_finite(g)) for g in grads]) if use_fp16 else tf.constant(True, dtype=tf.bool)

        # This is how the model was pre-trained.
        # ensure global norm is a finite number
        # to prevent clip_by_global_norm from having a hizzy fit.
        (clipped_grads, _) = tf.clip_by_global_norm(
            grads, clip_norm=clip_norm,
            use_norm=tf.cond(
                all_are_finite,
                lambda: tf.global_norm(grads),
                lambda: tf.constant(clip_norm)))

        train_op = optimizer.apply_gradients(
            list(zip(clipped_grads, tvars)), global_step=self.global_step)

        # Normally the global step update is done inside of `apply_gradients`.
        # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
        # a different optimizer, you should probably take this line out.
        new_global_step = tf.cond(all_are_finite, lambda: self.global_step + 1, lambda: self.global_step)
        new_global_step = tf.identity(new_global_step, name='step_update')
        self.train_op = tf.group(train_op, [self.global_step.assign(new_global_step)])


class AdamWeightDecayOptimizer(tf.train.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 name="AdamWeightDecayOptimizer"):
        """Constructs a AdamWeightDecayOptimizer."""
        super(AdamWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = tf.identity(learning_rate, name='learning_rate')
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            m = tf.get_variable(
                name=param_name + "/adam_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            v = tf.get_variable(
                name=param_name + "/adam_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())

            # Standard Adam update.
            next_m = (
                    tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
            next_v = (
                    tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                              tf.square(grad)))

            update = next_m / (tf.sqrt(next_v) + self.epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param

            update_with_lr = self.learning_rate * update

            next_param = param - update_with_lr

            assignments.extend(
                [param.assign(next_param),
                 m.assign(next_m),
                 v.assign(next_v)])
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name
