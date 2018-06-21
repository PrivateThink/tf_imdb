import data_helpers
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import time
import os
import datetime


# 加载词向量
def loadglove(filename):
    vocab = []
    embd = []
    file = open(filename,'r',encoding="utf8")
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('Loaded GloVe!')
    file.close()
    return vocab,embd

# 参数
neg_dir="data/rt-polaritydata/rt-polarity.neg"
pos_dir="data/rt-polaritydata/rt-polarity.pos"
word2vec="data/glove.6B.300d.txt"
# 句子长度
max_len=100
# 类别
num_classes=2
learning_rate=0.1
num_checkpoints=5
batch_size=64
num_epochs=100
display_every=10
evaluate_every=100
checkpoint_every=100
hidden_size=256
l2_reg_lambda=3

# 词向量加载
vocab, embd = loadglove(word2vec)
# 词典大小
vocab_size = len(vocab)
# 词向量维度大小
embedding_dim = len(embd[0])
embedding = np.asarray(embd)
del embd

with tf.device('/cpu:0'):
    x_text, y = data_helpers.load_data_and_labels(pos_dir,neg_dir)
#init vocab processor
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_len)
#fit the vocab from glove
pretrain = vocab_processor.fit(vocab)
#transform inputs
x = np.array(list(vocab_processor.transform(x_text)))
print("字典的大小：{:d}".format(len(vocab_processor.vocabulary_)))
print("x的大小", x.shape)
print("y的大小", y.shape)


# 按照8:2分割训练集和测试集
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,shuffle=True)
print("训练集/测试集: {:d}/{:d}\n".format(len(x_train), len(x_test)))

# 创建模型
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # 输入、输出、dropout占位符
        input_text=tf.placeholder(tf.int32,shape=[None,x_train.shape[1]],name='input_text')
        input_y=tf.placeholder(tf.float32,shape=[None,y_train.shape[1]], name='input_y')
        dropout_keep_prob=tf.placeholder(tf.float32,name='dropout_keep_prob')

        # 创建Embedding层
        with tf.name_scope("text-embedding"):
            W_text = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),trainable=False, name="W_text")
            embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
            embedding_init = W_text.assign(embedding_placeholder)
            embedded_chars = tf.nn.embedding_lookup(W_text, input_text)

            # 根据第二维展开,维度从0开始
            # 删除所有大小为1的维度,删除[1]为要删除维度的参数
            embedded_chars = tf.split(embedded_chars, max_len, 1)
            embedded_chars = [tf.squeeze(input_, [1]) for input_ in embedded_chars]

        with tf.name_scope("rnn"):
            # 前向cell和后向cell，分别加了dropout
            encoder_fw_cell = tf.contrib.rnn.LSTMCell(hidden_size)
            encoder_fw_cell = tf.nn.rnn_cell.DropoutWrapper(encoder_fw_cell, output_keep_prob=0.5)
            encoder_bw_cell = tf.contrib.rnn.LSTMCell(hidden_size)
            encoder_bw_cell = tf.nn.rnn_cell.DropoutWrapper(encoder_bw_cell, output_keep_prob=0.5)
            encoder_outputs, _,_ = tf.nn.static_bidirectional_rnn(
                encoder_fw_cell, encoder_bw_cell, embedded_chars,dtype=tf.float32
            )

        l2_loss = tf.constant(0.0)
        with tf.name_scope("output"):
            # 全连接层
            W = tf.get_variable("W", shape=[2*hidden_size, num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            logits_ = tf.nn.xw_plus_b(encoder_outputs[-1], W, b, name="logits")
            predictions = tf.argmax(logits_, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits_, labels=input_y)
            loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

            # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(predictions, tf.argmax(input_y, axis=1))
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")

        # 定义global_step和优化器
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

        # 保存模型
        timestamp=str(int(time.time()))
        out_dir=os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("模型保存的文件：",out_dir)

        loss_summary=tf.summary.scalar("loss",loss)
        acc_summary=tf.summary.scalar("accuracy",accuracy)
        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "text_vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        '''
        After creating a session and initialize global variables,
         run the embedding_init operation by feeding in the 2-D array embedding.
        '''
        sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), batch_size, num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            # Train
            feed_dict = {
                input_text: x_batch,
                input_y: y_batch,
                dropout_keep_prob: 0.5
            }
            _, step, summaries, train_loss, train_accuracy = sess.run(
                [train_op, global_step, train_summary_op, loss, accuracy], feed_dict)
            train_summary_writer.add_summary(summaries, step)

            # Training log display
            if step % display_every == 0:
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, train_loss, train_accuracy))

            # Evaluation
            if step % evaluate_every == 0:
                print("\nEvaluation:")
                feed_dict_dev = {
                    input_text: x_test,
                    input_y: y_test,
                    dropout_keep_prob: 1.0
                }
                summaries_dev, dev_loss, dev_accuracy = sess.run(
                    [dev_summary_op,loss, accuracy], feed_dict_dev)
                dev_summary_writer.add_summary(summaries_dev, step)

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}\n".format(time_str, step, dev_loss, dev_accuracy))

            # Model checkpoint
            if step % checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=step)
                print("Saved model checkpoint to {}\n".format(path))