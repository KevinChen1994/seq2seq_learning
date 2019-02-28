# seq2seq_learning
seq2seq learning with TensorFlow

我本地的环境是：
python=3.6
TensorFLow=1.12

使用TensorFlow实现了两个seq2seq，第一个simple的是简单的，只使用了双向的RNN作为编码器，单向RNN作为解码器。第二个common使用了teacher forcing，attention，beam search等技巧，效果相对好一点。

其中训练的参数没有做过多调整，训练时可以根据训练数据做一些相应调整。

训练的语料为清华大学的一个很小的机器翻译语料，说实话质量有点低，导致训练效果不是很好。

这个项目就当是自己学习TensorFlow的seq2seq api接口的练手项目吧。
