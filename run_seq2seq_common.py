# -*- coding:utf-8 -*-
# author:chenmeng
# datetime:2019/2/17 14:08
# software: PyCharm


import os
import time
import random
from collections import Counter

import tensorflow as tf

from model_seq2seq_common import seq2seq

source_dir = './data_mt/source.txt'
target_dir = './data_mt/target.txt'
source_vocab_dir = './data_mt/source_vocab.txt'
target_vocab_dir = './data_mt/target_vocab.txt'
source_validation_dir = './data_mt/source_validation.txt'
target_validation_dir = './data_mt/target_validation.txt'


class Config(object):
    embedding_dim = 100
    hidden_dim = 50
    batch_size = 128
    epoch = 20
    learning_rate = 0.005
    source_vocab_size = 20000
    target_vocab_size = 20000


# 构建词表
def make_vocab(data_dir, vocab_dir):
    with open(data_dir, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        words = []
        for i in lines:
            words.extend(i.replace('\n', '').split(' '))
        counter = Counter(words)
        result = counter.most_common(19997)
        result = [("_PAD", 0), ("_GO", 0), ("_EOS", 0), ("_UNKNOWN", 0)] + result
        with open(vocab_dir, 'w', encoding='utf-8') as f:
            for i in result:
                f.write(i[0])
                f.write('\n')


# 获得词与ID之间的字典
def word_to_id(vocab_dir):
    with open(vocab_dir, 'r', encoding='utf-8') as f:
        words = f.readlines()
        new_words = []
        for i in words:
            new_words.append(i.replace('\n', ''))
        word2id = dict(zip(new_words, range(len(new_words))))
        id2word = dict(zip(range(len(new_words)), new_words))
        return word2id, id2word


# 将文件转成id形式
def process_file(data_dir, word2id):
    doc_id = []
    with open(data_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            doc_id.append([word2id[x] if x in word2id else 3 for x in lines[i].replace('\n', '')])
    return doc_id


# 训练数据迭代器
def data_batch(source_id, target_id, source_word2id, target_word2id):
    data_len = len(source_id)
    n_batch = int(data_len // config.batch_size) + 1
    for b in range(n_batch):
        start_id = b * config.batch_size
        end_id = min((b + 1) * config.batch_size, data_len)
        max_source_len = max([len(sentence) for sentence in source_id[start_id: end_id]])
        max_target_len = max([len(sentence) for sentence in target_id[start_id: end_id]])
        source_len = [max_source_len if len(p) > max_source_len else len(p) for p in source_id[start_id: end_id]]
        target_len = [max_target_len if len(p) > max_target_len else len(p) for p in target_id[start_id: end_id]]

        source, target = [], []
        for i in source_id[start_id: end_id]:
            if len(i) >= max_source_len:
                source.append(i[0:max_source_len])
            else:
                source.append(i + [source_word2id["_PAD"]] * (max_source_len - len(i)))
        for i in target_id[start_id: end_id]:
            if len(i) >= max_target_len:
                target.append(i[0:max_target_len - 1] + [target_word2id["_EOS"]])
            else:
                target.append(i + [target_word2id["_PAD"]] * (max_target_len - 1 - len(i)) + [target_word2id["_EOS"]])
        yield (source, source_len, target, target_len)


# 生成验证集的一个batch
def data_batch_validation(source_id, target_id, source_word2id, target_word2id):
    data_len = len(source_id)
    start_id = random.randint(0, data_len - 129)
    end_id = start_id + 128
    max_source_len = max([50 if len(sentence) > 50 else len(sentence) for sentence in source_id[start_id: end_id]])
    max_target_len = max([50 if len(sentence) > 50 else len(sentence) for sentence in target_id[start_id: end_id]])
    source_len = [max_source_len if len(p) > max_source_len else len(p) for p in source_id[start_id: end_id]]
    target_len = [max_target_len if len(p) > max_target_len else len(p) for p in target_id[start_id: end_id]]
    source, target = [], []
    for i in source_id[start_id: end_id]:
        if len(i) >= max_source_len:
            source.append(i[0:max_source_len])
        else:
            source.append(i + [source_word2id["_PAD"]] * (max_source_len - len(i)))
    for i in target_id[start_id: end_id]:
        if len(i) >= max_target_len:
            target.append(i[0:max_target_len - 1] + [target_word2id["_EOS"]])
        else:
            target.append(i + [target_word2id["_PAD"]] * (max_target_len - 1 - len(i)) + [target_word2id["_EOS"]])
    return source, source_len, target, target_len


if __name__ == '__main__':
    print('make vocab table...')
    if not os.path.exists(source_vocab_dir):  # 如果不存在词汇表，重建
        make_vocab(source_dir, source_vocab_dir)
    if not os.path.exists(target_vocab_dir):  # 如果不存在词汇表，重建
        make_vocab(target_dir, target_vocab_dir)

    source_word2id, source_id2word = word_to_id(source_vocab_dir)
    target_word2id, target_id2word = word_to_id(target_vocab_dir)

    source_id = process_file(source_dir, source_word2id)
    target_id = process_file(target_dir, target_word2id)
    source_validation_id = process_file(source_validation_dir, source_word2id)
    target_validation_id = process_file(target_validation_dir, target_word2id)

    print('build model...')
    config = Config()
    model = seq2seq(config, source_word2id, tearcherForcing=True, attention=True, beamSearch=1)

    print('training...')
    with tf.Session() as sess:
        tf.summary.FileWriter('graph', sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        print_every = 100
        n_batch = int(len(source_id) // config.batch_size) + 1
        for e in range(config.epoch):
            batch = 0
            for source, source_len, target, target_len in data_batch(source_id, target_id, source_word2id,
                                                                     target_word2id):
                batch += 1
                feed_dict = {
                    model.batch_size: len(source),
                    model.seq_inputs: source,
                    model.seq_inputs_length: source_len,
                    model.seq_targets: target,
                    model.seq_targets_length: target_len
                }
                # 训练集损失
                loss, _ = sess.run([model.loss, model.train_op], feed_dict)

                if batch % print_every == 0 and batch > 0:
                    source_validation, source_len_validation, target_validation, target_len_validation = data_batch_validation(
                        source_validation_id, target_validation_id, source_word2id,
                        target_word2id)
                    feed_dict_validation = {
                        model.batch_size: len(source_validation),
                        model.seq_inputs: source_validation,
                        model.seq_inputs_length: source_len_validation,
                        model.seq_targets: target_validation,
                        model.seq_targets_length: target_len_validation
                    }
                    # 验证集损失
                    loss_validation = sess.run(model.loss, feed_dict_validation)

                    print("-----------------------------")
                    print("epoch:", e)
                    print("batch:", batch, "/", n_batch)
                    print("time:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
                    print("loss_train:", loss)
                    print("loss_validation:", loss_validation)

        print(saver.save(sess, "checkpoint/common/model.ckpt"))
