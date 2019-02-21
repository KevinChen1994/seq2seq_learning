# -*- coding:utf-8 -*-
# author:chenmeng
# datetime:2019/2/14 16:11
# software: PyCharm

import tensorflow as tf


class seq2seq(object):
    def build_inputs(self, config):
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch_size')
        self.seq_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='seq_inputs')
        self.seq_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='seq_inputs_length')
        self.seq_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='seq_targets')
        self.seq_targets_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='seq_targets_length')

    def __init__(self, config, w2i_target):
        self.build_inputs(config)
        with tf.variable_scope('encoder'):
            encoder_embedding = tf.Variable(tf.random_uniform([config.source_vocab_size, config.embedding_dim]),
                                            dtype=tf.float32, name='encoder_embedding')
            encoder_inputs_embedded = tf.nn.embedding_lookup(encoder_embedding, self.seq_inputs)

            with tf.variable_scope("gru_cell"):
                encoder_cell = tf.nn.rnn_cell.GRUCell(config.hidden_dim)

            ((encoder_fw_outputs, encoder_bw_outputs),
             (encoder_fw_final_state, encoder_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                                                                                 cell_bw=encoder_cell,
                                                                                                 inputs=encoder_inputs_embedded,
                                                                                                 sequence_length=self.seq_inputs_length,
                                                                                                 dtype=tf.float32,
                                                                                                 time_major=False)
            encoder_state = tf.add(encoder_fw_final_state, encoder_bw_final_state)
            encoder_outputs = tf.add(encoder_fw_outputs, encoder_bw_outputs)

        with tf.variable_scope('decoder'):
            decoder_embedding = tf.Variable(tf.random_normal([config.target_vocab_size, config.embedding_dim]),
                                            dtype=tf.float32, name='decoder_embedding')
            token_go = tf.ones([self.batch_size], dtype=tf.int32, name='token_go') * w2i_target['_GO']

            # helper对象
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embedding, token_go, w2i_target["_EOS"])

            with tf.variable_scope('gru_cell'):
                decoder_cell = tf.nn.rnn_cell.GRUCell(config.hidden_dim)

                decoder_initial_state = encoder_state

            # 构建decoder
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, decoder_initial_state,
                                                      output_layer=tf.layers.Dense(config.target_vocab_size))
            decoder_outputs, decoder_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                                                       maximum_iterations=tf.reduce_max(
                                                                                                           self.seq_targets_length))

            self.decoder_logits = decoder_outputs.rnn_output
            self.out = tf.argmax(self.decoder_logits, 2)

            # mask掉填充的0，使后边计算的时候0不参与计算。
            sequence_mask = tf.sequence_mask(self.seq_targets_length, dtype=tf.float32)
            self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.decoder_logits, targets=self.seq_targets,
                                                         weights=sequence_mask)
            opt = tf.train.AdamOptimizer(config.learning_rate)
            gradients = opt.compute_gradients(self.loss)
            capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
            self.train_op = opt.apply_gradients(capped_gradients)
