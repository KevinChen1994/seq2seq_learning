# -*- coding:utf-8 -*-
# author:chenmeng
# datetime:2019/2/17 14:08
# software: PyCharm

import tensorflow as tf
from tensorflow.contrib import seq2seq as seq2seq_contrib


class seq2seq(object):
    def build_inputs(self, config):
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch_size')
        self.seq_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='seq_inputs')
        self.seq_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='seq_inputs_length')
        self.seq_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='seq_targets')
        self.seq_targets_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='seq_targets_length')

    def __init__(self, config, w2i_target, train=True, attention=True, beamSearch=1):
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
            if train:
                # 在target前加上GO标签，并且去掉最后一个字符<EOS>，因为这个字符是不参数decoder的输入的
                decoder_inputs = tf.concat([tf.reshape(token_go, [-1, 1]), self.seq_targets[:, :-1]], 1)
                helper = seq2seq_contrib.TrainingHelper(tf.nn.embedding_lookup(decoder_embedding, decoder_inputs),
                                                self.seq_targets_length)
            else:
                helper = seq2seq_contrib.GreedyEmbeddingHelper(decoder_embedding, token_go, w2i_target["_EOS"])

            with tf.variable_scope('gru_cell'):
                decoder_cell = tf.nn.rnn_cell.GRUCell(config.hidden_dim)

                if attention:
                    if beamSearch > 1:
                        tiled_encoder_outputs = seq2seq_contrib.tile_batch(encoder_outputs, multiplier=beamSearch)
                        tiled_sequence_length = seq2seq_contrib.tile_batch(self.seq_inputs_length, multiplier=beamSearch)
                        attention_mechanism = seq2seq_contrib.BahdanauAttention(num_units=config.hidden_dim,
                                                                        memory=tiled_encoder_outputs,
                                                                        memory_sequence_length=tiled_sequence_length)
                        decoder_cell = seq2seq_contrib.AttentionWrapper(cell=decoder_cell,
                                                                attention_mechanism=attention_mechanism)
                        tiled_encoder_final_state = seq2seq_contrib.tile_batch(encoder_state, multiplier=beamSearch)
                        tiled_decoder_initial_state = decoder_cell.zero_state(batch_size=self.batch_size * beamSearch,
                                                                        dtype=tf.float32)
                        tiled_decoder_initial_state = tiled_decoder_initial_state.clone(cell_state=tiled_encoder_final_state)
                        decoder_initial_state = tiled_decoder_initial_state
                    else:
                        attention_mechanism = seq2seq_contrib.BahdanauAttention(num_units=config.hidden_dim,
                                                                        memory=encoder_outputs,
                                                                        memory_sequence_length=self.seq_inputs_length)
                        # 另外一种attention计算方式
                        # attention_mechanism = seq2seq_contrib.LuongAttention(num_units=config.hidden_dim,
                        #                                                 memory=encoder_outputs,
                        #                                                 memory_sequence_length=self.seq_inputs_length)
                        decoder_cell = seq2seq_contrib.AttentionWrapper(cell=decoder_cell,
                                                                attention_mechanism=attention_mechanism)
                        decoder_initial_state = decoder_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
                        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)

                else:
                    if beamSearch > 1:
                        decoder_initial_state = seq2seq_contrib.tile_batch(encoder_state, multiplier=beamSearch)
                    else:
                        decoder_initial_state = encoder_state

            if beamSearch > 1:
                # 构建decoder
                decoder = seq2seq_contrib.BeamSearchDecoder(cell=decoder_cell, embedding=decoder_embedding,
                                                    start_tokens=token_go,
                                                    end_token=w2i_target['_EOS'], initial_state=decoder_initial_state,
                                                    beam_width=beamSearch,
                                                    output_layer=tf.layers.Dense(units=config.target_vocab_size))
            else:
                # 构建decoder
                decoder = seq2seq_contrib.BasicDecoder(decoder_cell, helper, decoder_initial_state,
                                               output_layer=tf.layers.Dense(config.target_vocab_size))
            decoder_outputs, decoder_state, final_sequence_lengths = seq2seq_contrib.dynamic_decode(decoder,
                                                                                            maximum_iterations=tf.reduce_max(
                                                                                                self.seq_targets_length))
        if beamSearch > 1:
            self.out = decoder_outputs.predicted_ids[:, :, 0]
        else:
            self.decoder_logits = decoder_outputs.rnn_output
            self.out = tf.argmax(self.decoder_logits, 2)

            # mask掉填充的0，使后边计算的时候0不参与计算。
            sequence_mask = tf.sequence_mask(self.seq_targets_length, dtype=tf.float32)
            self.loss = seq2seq_contrib.sequence_loss(logits=self.decoder_logits, targets=self.seq_targets,
                                              weights=sequence_mask)
            opt = tf.train.AdamOptimizer(config.learning_rate)
            gradients = opt.compute_gradients(self.loss)
            capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
            self.train_op = opt.apply_gradients(capped_gradients)
