# -*- coding:utf-8 -*-
# author:chenmeng
# datetime:2019/2/15 16:34
# software: PyCharm
import tensorflow as tf
import thulac

from model_seq2seq_simple import seq2seq
from run_seq2seq_simple import word_to_id, Config

thul = thulac.thulac(seg_only=True)

source_dir = './data_mt/source.txt'
target_dir = './data_mt/target.txt'
source_vocab_dir = './data_mt/source_vocab.txt'
target_vocab_dir = './data_mt/target_vocab.txt'
model_path = 'checkpoint/simple/model.ckpt'

if __name__ == '__main__':
    print('load data...')
    source_word2id, source_id2word = word_to_id(source_vocab_dir)
    target_word2id, target_id2word = word_to_id(target_vocab_dir)

    print('build model...')
    config = Config()
    model = seq2seq(config, source_word2id)

    print('run model...')
    max_source_length = 50
    max_target_length = 50

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        source_raw = '经济、社会和环境方面的发展、人居议程、森林、能源、水和卫生设施'
        source_cut = thul.cut(source_raw, text=True)

        source_id = [source_word2id[word] for word in source_cut.split(' ') if word in source_word2id]

        source_len = [len(source_id)]

        source, target = [], []

        if source_len[0] >= max_source_length:
            source.append(source_id[0:max_source_length])
        else:
            source.append(source_id + [source_word2id["_PAD"]] * (max_source_length - source_len[0]))

        target_len = [max_target_length]
        target = [[0] * max_target_length]

        feed_dict = {
            model.batch_size: len(source),
            model.seq_inputs: source,
            model.seq_inputs_length: source_len,
            model.seq_targets: target,
            model.seq_targets_length: target_len
        }

        predict = sess.run(model.out, feed_dict)

        print('in:', source_raw)
        print('out:', [target_id2word[i] for i in predict[0] if target_id2word[i] != '_PAD'])
