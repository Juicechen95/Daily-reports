# -*- coding: utf-8 -*-
import os, sys, re
import pickle as cPickle
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import scipy.io as sio
from math import floor
from sklearn.metrics import roc_auc_score


import argparse
import logging
from test import *

def recall_disturb(prob, threshold, labels):
    disturb = 0
    recall = 0
    for i in range(prob.__len__()):
        if prob[i] / 3.0 >= threshold:
            if labels[i] == 0:
                disturb += 1
            else:
                recall += 1
            
    return recall * 1.0 / sum(labels), disturb * 1.0 / (len(labels) - sum(labels))

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a prunned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    prunned so subgraphs that are not neccesary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

def chineseidx_process(input_list, wordtoix, ixtoword):
    re_words = re.compile(u'[\u4e00-\u9fa5]+')
    res_list = []
    for (i, x) in enumerate(input_list):
        res_list.append([wordtoix[tmp_word] for tmp_word in re_words.findall(','.join([ixtoword[y] for y in x])) if tmp_word in wordtoix])
    return res_list

def save_model(saver, sess, output_name, save_dir='./model/'):
    from tensorflow.python.tools import freeze_graph

    saver.save(sess, save_dir + "data-all")
    tf.train.write_graph(sess.graph_def, save_dir, "model.txt", True)

    checkpoint = tf.train.get_checkpoint_state(save_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    input_graph = save_dir + "model.txt"
    output_graph = save_dir + "frozen_model.pb"
    freeze_graph.freeze_graph(input_graph=input_graph,
                      input_saver="",
                      input_checkpoint=input_checkpoint,
                      output_graph=output_graph,
                      input_binary=False,
                      output_node_names=output_name.split(':')[0],  # change this to match your graph
                      restore_op_name="save/restore_all",
                      filename_tensor_name="save/Const:0",
                      clear_devices=True,
                      initializer_nodes="")

    with tf.gfile.GFile(output_graph, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.train.write_graph(graph_def, save_dir, 'frozen_graph.pbtxt', False)
    print('Saved')
        
class Options(object):
    def __init__(self):
        self.GPUID = 0
        self.dataset = 'agnews'
        self.fix_emb = False
        self.restore = False
        self.W_emb = None
        self.W_class_emb = None
        self.maxlen = 305
        self.n_words = None
        self.embed_size = 300
        self.lr = 1e-4
        self.max_epochs = 5
        self.dropout = 0.5
        self.part_data = False
        self.portion = 1
        self.save_path = "./save/"
        self.log_path = "./log/"
        self.valid_freq = 5
        self.batch_size = 2
        self.optimizer = 'Adam'
        self.clip_grad = None
        self.class_penalty = 1.0
        self.H_dis = 300
        self.num_none = 3
        self.num_att = 4
        self.latent_size = 300
        self.att_type = 'w'  # cos/w
        self.co_att_type_xy = 'cos'  # cos/w
        self.att_none_linear = False
        self.att_pooling = 'max_conv'
        self.ngram = 55
        self.model = '1234'
        self.kernels = 4
        self.att_w = tf.get_variable('att_w', [self.embed_size, self.embed_size])
        self.inter_class_penalty = 1.0
        self.intra_class_penalty = 1.0
        self.wordlabel_ratio = 0
        self.multigram_leam=False
        self.feature_aggr='sum'
        self.leam_ngram = 55
        self.ensemble = True
        self.save_feature = True
        self.save_size = 1000


def unified_classifier(x, x_mask, y, dropout, opt, class_penalty):
    #  comment notation
    #  b: batch size, s: sequence length, e: embedding dim, c : num of class
    x_emb, W_norm = embedding(x, opt)  # b * s * e
    print('tlfalg', x_emb.get_shape(), x_emb.name, W_norm.get_shape(), W_norm.name)
    x_emb = tf.cast(x_emb, tf.float32)
    W_norm = tf.cast(W_norm, tf.float32)
    y_pos = tf.argmax(y, -1)
    y_emb, W_class = embedding_class(y_pos, opt, 'class_emb')  # b * e, c * e
    W_class = tf.cast(W_class, tf.float32)
    W_class_tran = tf.transpose(W_class, [1, 0])  # e * c

    x_multigram = []
    x_multigram.append(tf.multiply(x_emb, tf.expand_dims(x_mask, -1)))
    for ngram in opt.ngram:
        x_conv = tf.contrib.layers.conv2d(tf.expand_dims(tf.multiply(x_emb, tf.expand_dims(x_mask, -1)), -1), num_outputs=opt.kernels,
                                          kernel_size=[ngram, 1],
                                          padding='SAME',
                                          activation_fn=tf.nn.relu,
                                          scope='conv_%dgram' % ngram)
        tmp_x0 = tf.reduce_max(x_conv, axis=-1, keep_dims=True)
        tmp_xmask = tf.expand_dims(x_mask, -1)
        print('tmp_x_conv0', tmp_x0.get_shape(), 'mask', tmp_xmask.get_shape())
        tmp_x = tf.multiply(tf.squeeze(tmp_x0, axis=-1), tf.expand_dims(x_mask, -1))
        print('tmp_x_conv', tmp_x.get_shape())
        x_multigram.append(tf.multiply(tf.squeeze(tf.reduce_max(x_conv, axis=-1, keep_dims=True), axis=-1), tf.expand_dims(x_mask, -1)))

    x_self_att, x_self_att_v, x_self_att_score = self_att(tf.concat(x_multigram, axis=1, name='x_concat'),
                          tf.concat([x_mask] * len(x_multigram), axis=1), opt, 'x_')  # x0: b * e

    x_co_att, x_co_att_score = co_att_xy_multigram(x_multigram, x_mask, W_class, opt)  # list[xy: b * s * e] * ngram

    x_concat = tf.concat(x_multigram + x_co_att, axis=1, name='x_xy_concat')  # b * 2ns * e
    x_concat_mask = tf.concat([x_mask] * 2 * len(x_multigram), axis=1)

    x_concat_self_att, x_concat_self_att_v, x_concat_self_att_score = self_att(x_concat, x_concat_mask, opt, 'x_concat_')  # x1: b * e
    x_co_att_self_co_att, x_co_att_self_co_att_score = self_co_att_pooling_mask_multigram(x_co_att, x_mask, opt, 'xy')  # x2: b * e
    x_self_co_att, x_self_co_att_score = self_co_att_pooling_mask_multigram(x_multigram, x_mask, opt, 'xx')  # x3: b * e
    if opt.multigram_leam:
        x_leam, x_leam_score = att_emb_ngram_encoder_maxout_multigram(x_multigram, x_mask, W_class, W_class_tran, opt)  # x4: b * e
    else:
        x_leam, x_leam_score = att_emb_ngram_encoder_maxout(x_emb, x_mask, W_class, W_class_tran, opt)
    if opt.feature_aggr == "sum":
        feature = tf.zeros_like(tf.squeeze(x_leam))
        if '0' in opt.model:
            feature += x_self_att
        if '1' in opt.model:
            feature += x_concat_self_att
        if '2' in opt.model:
            feature += x_co_att_self_co_att
        if '3' in opt.model:
            feature += x_self_co_att
        if '4' in opt.model:
            feature += tf.squeeze(x_leam)
    elif opt.feature_aggr == "self":
        feature_list = []
        if '0' in opt.model:
            feature_list.append(x_self_att)
        if '1' in opt.model:
            feature_list.append(x_concat_self_att)
        if '2' in opt.model:
            feature_list.append(x_co_att_self_co_att)
        if '3' in opt.model:
            feature_list.append(x_self_co_att)
        if '4' in opt.model:
            feature_list.append(tf.squeeze(x_leam))
        feature, feature_self_att_v, feature_self_att_score = self_att(tf.stack(feature_list, axis=1), tf.ones([opt.batch_size, len(feature_list)]), opt, 'feature_')
    elif opt.feature_aggr == "mlp":
        feature_list = []
        if '0' in opt.model:
            feature_list.append(x_self_att)
        if '1' in opt.model:
            feature_list.append(x_concat_self_att)
        if '2' in opt.model:
            feature_list.append(x_co_att_self_co_att)
        if '3' in opt.model:
            feature_list.append(x_self_co_att)
        if '4' in opt.model:
            feature_list.append(tf.squeeze(x_leam))
        feature = tf.squeeze(tf.contrib.layers.fully_connected(inputs=tf.stack(feature_list, axis=2), num_output=1,
                                                               activation_fn=tf.nn.leaky_relu))
    else:
        raise ValueError('No feature aggregation method named:%s' % opt.feature_aggr)

    intra_class_loss = 0
    W_wordlabel = []
    for y_i in range(opt.num_class):
        tmpx = tf.where(tf.equal(y_pos, y_i), feature, tf.zeros_like(feature))
        if tf.reduce_sum(tmpx) == 0:
            intra_class_loss += 0
            W_wordlabel.append(tf.zeros([opt.embed_size]))
        else:
            mean_x = tf.reduce_mean(tmpx, axis=0)
            conv_x = tf.matmul(tmpx - mean_x, tmpx - mean_x, transpose_a=True)
            intra_class_loss += tf.trace(conv_x)
            W_wordlabel.append(mean_x)


    if opt.ensemble:
        print('ensemble')
        logits_x_self = discriminator_2layer(x_self_att, opt, dropout, prefix='classify_x_self_',
                                           num_outputs=opt.num_class,
                                           is_reuse=False)
        logits_self = discriminator_2layer(x_concat_self_att, opt, dropout, prefix='classify_self_', num_outputs=opt.num_class,
                                      is_reuse=False)
        logits_xx = discriminator_2layer(x_self_co_att, opt, dropout, prefix='classify_xx_',
                                           num_outputs=opt.num_class,
                                           is_reuse=False)
        print('flag2', x_leam.get_shape(), tf.squeeze(x_leam).get_shape())
        logits_xy = discriminator_2layer(x_leam, opt, dropout, prefix='classify_xy_',
                                           num_outputs=opt.num_class,
                                           is_reuse=False)
        logits_class = discriminator_2layer(W_class, opt, dropout, prefix='classify_xy_', num_outputs=opt.num_class,
                                            is_reuse=True)

        prob_x_self = tf.nn.softmax(logits_x_self, name='prob_x_self')
        feature_loss_x_self = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits_x_self))
        prob_self = tf.nn.softmax(logits_self, name='prob_self')
        feature_loss_self = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits_self))
        prob_xx = tf.nn.softmax(logits_xx, name='prob_xx')
        feature_loss_xx = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits_xx))
        prob_xy = tf.nn.softmax(logits_xy, name='prob_xy')
        feature_loss_xy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits_xy))

        prob = tf.zeros_like(prob_self, name='prob')
        print('prob_first_name', prob.name)
        loss = tf.zeros_like(feature_loss_self)
        if '0' in opt.model:
            prob = tf.add(prob, prob_x_self, name='prob')
            loss += feature_loss_x_self
        if '1' in opt.model:
            prob = tf.add(prob, prob_self, name='prob')
            loss += feature_loss_self
        if '3' in opt.model:
            prob = tf.add(prob, prob_xx, name='prob')
            loss += feature_loss_xx
        if '4' in opt.model:
            prob = tf.add(prob, prob_xy, name='prob')
            loss += feature_loss_xy
        predictions = tf.argmax(prob, 1, name='predictions')
        class_y = tf.constant(name='class_y', shape=[opt.num_class, opt.num_class], dtype=tf.float32,
                              value=np.identity(opt.num_class), )
        correct_prediction = tf.equal(tf.argmax(prob, 1), tf.argmax(y, 1), name='correct_prediction')
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_')
        class_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=class_y, logits=logits_class))
        W_class_norm = tf.nn.l2_normalize(W_class, axis=1)
        inter_class_loss = tf.losses.mean_squared_error(class_y,
                                                        tf.matmul(W_class_norm, W_class_norm, transpose_b=True))

        loss += class_penalty * class_loss
        if opt.inter_class_penalty > 0:
            loss = loss + opt.inter_class_penalty * inter_class_loss
        if opt.intra_class_penalty > 0:
            loss = loss + opt.intra_class_penalty * intra_class_loss

    else:
        print('unensemble')
        logits = discriminator_2layer(feature, opt, dropout, prefix='classify_', num_outputs=opt.num_class,
                                      is_reuse=False)
        logits_class = discriminator_2layer(W_class, opt, dropout, prefix='classify_', num_outputs=opt.num_class,
                                            is_reuse=True)

        prob = tf.nn.softmax(logits, name='prob')
        predictions = tf.argmax(prob, 1, name='predictions')
        class_y = tf.constant(name='class_y', shape=[opt.num_class, opt.num_class], dtype=tf.float32,
                              value=np.identity(opt.num_class), )
        correct_prediction = tf.equal(tf.argmax(prob, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_')

        feature_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
        class_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=class_y, logits=logits_class))

        W_class_norm = tf.nn.l2_normalize(W_class, axis=1)
        inter_class_loss = tf.losses.mean_squared_error(class_y,
                                                        tf.matmul(W_class_norm, W_class_norm, transpose_b=True))

        loss = feature_loss + class_penalty * class_loss + opt.inter_class_penalty * inter_class_loss + opt.intra_class_penalty * intra_class_loss

    global_step = tf.Variable(0, trainable=False)
    train_op = layers.optimize_loss(
        loss,
        global_step=global_step,
        optimizer=opt.optimizer,
        learning_rate=opt.lr)
    W_class = (1 - opt.wordlabel_ratio) * W_class + opt.wordlabel_ratio * tf.stack(W_wordlabel)
    W_class_update = update_label_embedding(W_class, 'class_emb')

    kernels = []
    for ngram in opt.ngram:
        tmp_kernel = tf.squeeze(tf.get_collection(tf.GraphKeys.VARIABLES, 'conv_%dgram/weights' % ngram)[0])
        kernels.append(tmp_kernel)
    
#    print('out_shape', predictions.get_shape(), prob.get_shape(), prob[:,1].get_shape())
#    predictions = tf.cast(prob[:,1], tf.float32, name='predictions')
    print('out_name', prob.name, predictions.name)
    return accuracy, loss, train_op, W_norm, global_step, W_class_update, x_self_att_v, x_self_att_score,\
           x_co_att_score, x_concat_self_att_v, x_concat_self_att_score, x_co_att_self_co_att_score,\
           x_self_co_att_score, x_leam_score, x_self_att, x_concat_self_att, x_co_att_self_co_att,\
           x_self_co_att, tf.squeeze(x_leam), feature, kernels, correct_prediction, predictions, prob


if __name__ == '__main__':
#    main()
    # Prepare training and testing data
    opt = Options()
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', action='store', dest="lr", default=1e-3, type=float,
                        help="learning rate. 1e-4(default)")
    parser.add_argument('-d', '--dataset', action='store', dest="dataset", default='web',
                        help="amazon_full/yahoo/yelp_full/dbpedia/agnews/web(default)")
    parser.add_argument('-i', '--intra_att', action='store', dest="intra_att", default='cos',
                        help="intra-attention type: share/mul/w/cos(default)")
    parser.add_argument('-c', '--co_att', action='store', dest="co_att", default='cos',
                        help="co-attention type: share/mul/w/cos(default)")
    parser.add_argument('-p', '--pooling', action='store', dest="pool", default='mean',
                        help="attention pooling type: max/mean(default)")
    parser.add_argument('-n', '--ngram', action='store', dest="ngram", default="25,55,75",
                        help="ngram hyper-parameter: 25,55(default)")
    parser.add_argument('-a', '--alpha', action='store', dest="alpha", default=1.0, type=float,
                        help="class penalty ratio: 1.0(default)")
    parser.add_argument('-inter', action='store', dest="inter_alpha", default=0.0, type=float,
                        help="inter-class penalty ratio: 0.0(default)")
    parser.add_argument('-intra', action='store', dest="intra_alpha", default=0.0, type=float,
                        help="intra-class penalty ratio: 0.0(default)")
    parser.add_argument('-w', action='store', dest="wordlabel_ratio", default=0, type=float,
                        help="update ratio for label embedding with mean of sequence representation: 0(default)")
    parser.add_argument('-e', action='store', dest="max_epoch", default=1, type=int,
                        help="max epoch: 5(default)")
    parser.add_argument('-k', action='store', dest="kernel", default=4, type=int,
                        help="output number of each kernels: 4(default)")
    parser.add_argument('-att_nonlinear', action='store', dest="att_nonlinear", default=False, type=bool,
                        help="attention matrix with nonlinear function(tanh): True/False(default)")
    parser.add_argument('-model', action='store', dest="model", default="134",
                        help="select model: 0(default). 0:self attention[x], 1:self attention[x,xy], 2:intra-co-attention[xy], 3:intra-attention[xx], 4:LEAM")
    parser.add_argument('-feature', action='store', dest="feature_aggr", default="sum",
                        help="feature aggregation method: sum(default)/self/mlp")
    parser.add_argument('-multi_leam', action='store', dest="multi_leam", default=True, type=bool,
                        help="Multi-gram leam (no conv): False/True(default)")
    parser.add_argument('-leam_ngram', action='store', dest="leam_ngram", default=55, type=int,
                        help="N-gram conv in leam: 55(default)")
    parser.add_argument('-ensemble', action='store', dest="ensemble", default=True, type=bool,
                        help="Instead of using feature vector aggregation, the model use 3 different classifier and sum up the three loss.")
    parser.add_argument('-save_feature', action='store', dest="save_feature", default=False, type=bool,
                        help="Save attention scores and features: False(default)/True")
    parser.add_argument('-save_path', action='store', dest="save_path", default='./Model/', type=str,
                    help="The path of the saver")

    args = parser.parse_args()
    d = args.__dict__
    opt.lr = d['lr']
    opt.dataset = d['dataset']
    opt.att_type = d['intra_att']
    opt.co_att_type_xy = d['co_att']
    opt.att_pooling = d['pool']
    opt.class_penalty = d['alpha']
    opt.att_none_linear = d['att_nonlinear']
    opt.model = d['model']
#    opt.max_epochs = d['max_epoch']
    opt.ngram = [int(i) for i in d['ngram'].split(',')]
    opt.kernels = d['kernel']
    opt.inter_class_penalty = d['inter_alpha']
    opt.intra_class_penalty = d['intra_alpha']
    opt.wordlabel_ratio = d['wordlabel_ratio']
    opt.feature_aggr = d['feature_aggr']
    opt.multigram_leam = d['multi_leam']
    opt.leam_ngram = d['leam_ngram']
    opt.ensemble = d['ensemble']
    opt.save_feature = d['save_feature']

    decay_dict = {}

    # load data
    if opt.dataset == 'yahoo':
        loadpath = "./data/yahoo.p"
        embpath = "./data/yahoo_glove.p"
        opt.num_class = 10
        opt.class_name = ['Society Culture',
                          'Science Mathematics',
                          'Health',
                          'Education Reference',
                          'Computers Internet',
                          'Sports',
                          'Business Finance',
                          'Entertainment Music',
                          'Family Relationships',
                          'Politics Government']
        decay_dict['yahoo'] = 0.76
    elif opt.dataset == 'agnews':
        loadpath = "./data/ag_news.p"
        embpath = "./data/ag_news_glove.p"
        opt.num_class = 4
        opt.class_name = ['World Election State President Police Politics Security War Nuclear Democracy Militant',
                          'Sports Olympic Football Sport League Baseball Rugby Tickets Basketball Games Championship',
                          'Business Company Market Oil Consumers Exchange Product Price Billion Stocks',
                          'Science Laboratory Computers Science Technology Web Google Microsoft Economy Software Investment']
        decay_dict['agnews'] = 0.91
    elif opt.dataset == 'dbpedia':
        loadpath = "./data/dbpedia.p"
        embpath = "./data/dbpedia_glove.p"
        opt.num_class = 14
        opt.class_name = ['Company',
                          'Educational Institution',
                          'Artist',
                          'Athlete',
                          'Office Holder',
                          'Mean Of Transportation',
                          'Building',
                          'Natural Place',
                          'Village',
                          'Animal',
                          'Plant',
                          'Album',
                          'Film',
                          'Written Work'
                          ]
        decay_dict['dbpedia'] = 0.98
    elif opt.dataset == 'yelp_full':
        loadpath = "./data/yelp_full.p"
        embpath = "./data/yelp_full_glove.p"
        opt.num_class = 5
        opt.class_name = ['worst',
                          'bad',
                          'middle',
                          'good',
                          'best']
        decay_dict['yelp_full'] = 0.63
    elif opt.dataset == 'amazon_full':
        loadpath = "./data/amazon_full.p"
        embpath = "./data/amazon_full_glove.p"
        opt.num_class = 5
        opt.class_name = ['worst',
                          'bad',
                          'middle',
                          'good',
                          'best']
        decay_dict['amazon_full'] = 0.59
    elif opt.dataset == 'web':
        loadpath = "./data/web_gambling_10.p"
        embpath = "./data/web_gambling_emb_10.p"
        train_data_path = './data/web_train_data.p'
        opt.num_class = 2
        opt.class_name = ['正常',
                          '赌博 投注 博彩']
        decay_dict['web'] = 0.8
    x = cPickle.load(open(loadpath, "rb"), encoding='bytes')
#    [x_supp, x_supp_lab] = cPickle.load(open('./data/web_train_normal.p', 'rb'), encoding='bytes')
#    train, val, test = x[0], x[1], x[2]
#    train_lab, val_lab, test_lab = x[3], x[4], x[5]
    wordtoix, ixtoword = x[6], x[7]
    [x, y] = cPickle.load(open(train_data_path, "rb"), encoding='bytes')
    np.random.seed(0)
    shuffle_idx = np.random.choice(len(x), len(x), replace=False)
    train = [x[i] for i in shuffle_idx[:6000]]
    train_lab = [y[i] for i in shuffle_idx[:6000]]
    val = [x[i] for i in shuffle_idx[6000:8000]]
    val_lab = [y[i] for i in shuffle_idx[6000:8000]]
    test = [x[i] for i in shuffle_idx[8000:]]
    test_lab = [y[i] for i in shuffle_idx[8000:]]
#    train.extend(x_supp)
#    train_lab.extend(x_supp_lab)
    print(len(train), len(val), len(test))
    train = chineseidx_process(train, wordtoix, ixtoword)
    val = chineseidx_process(val, wordtoix, ixtoword)
    test = chineseidx_process(test, wordtoix, ixtoword)
    del x
    del y
    print("load data finished")

    if opt.dataset == 'amazon_full':
        train_lab_array = np.zeros([len(train_lab), opt.num_class], dtype='float32')
        val_lab_array = np.zeros([len(val_lab), opt.num_class], dtype='float32')
        test_lab_array = np.zeros([len(test_lab), opt.num_class], dtype='float32')
        for i in range(len(train_lab)):
            train_lab_array[i][int(train_lab[i])-1] = 1
        for i in range(len(val_lab)):
            val_lab_array[i][int(val_lab[i])-1] = 1
        for i in range(len(test_lab)):
            test_lab_array[i][int(test_lab[i])-1] = 1
        del train_lab
        del val_lab
        del test_lab
        train_lab = train_lab_array
        val_lab = val_lab_array
        test_lab = test_lab_array
    else:
        train_lab = np.array(train_lab, dtype='float32')
        val_lab = np.array(val_lab, dtype='float32')
        test_lab = np.array(test_lab, dtype='float32')
    opt.n_words = len(ixtoword)
    if opt.part_data:
        # np.random.seed(123)
        train_ind = np.random.choice(len(train), int(len(train) * opt.portion), replace=False)
        train = [train[t] for t in train_ind]
        train_lab = [train_lab[t] for t in train_ind]

    print('Total words: %d' % opt.n_words)

#    np.random.seed(0)
#    train_ind_save = np.random.choice(len(test), opt.save_size, replace=False)
#    train_save = []
#    train_lab_save = []
#    for wrong_idx in [8, 1242, 1508, 1767, 2157, 2221, 2254, 2458, 2510, 2520, 2534, 2557, 2607, 2765, 2812, 2846, 2860, 2961]:
#        train_save.append(test[wrong_idx])
#        train_lab_save.append(test_lab[wrong_idx])
#    for t in train_ind_save[:-18]:
#        train_save.append(test[t])
#        train_lab_save.append(test_lab[t])

    #train_save = [train[t] for t in train_ind_save]
    #train_lab_save = [train_lab[t] for t in train_ind_save]


    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.GPUID)

    try:
        if opt.dataset == 'amazon_full' or opt.dataset == 'web':
            print('load the trained embedding')
            opt.W_emb = np.array(cPickle.load(open(embpath, 'rb'), encoding='bytes'), dtype='float32')[0]
        else:
            opt.W_emb = np.array(cPickle.load(open(embpath, 'rb'), encoding='bytes'), dtype='float32')
        if opt.fix_emb:
            opt.W_class_emb = load_class_embedding(wordtoix, opt)
    except IOError:
        print('No embedding file found.')
        opt.fix_emb = False
        
    sess = tf.InteractiveSession(graph=tf.Graph())
    x_ = tf.placeholder(tf.int32, shape=[None, opt.maxlen], name='x_')
    x_mask_ = tf.placeholder(tf.float32, shape=[None, opt.maxlen], name='x_mask_')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    y_ = tf.placeholder(tf.float32, shape=[None, opt.num_class], name='y_')
    class_penalty_ = tf.placeholder(tf.float32, shape=(), name='class_penalty_')
    accuracy_, loss_, train_op, W_norm_, global_step, update_label, x0v, x0a, xya, x1v, x1a, x2a, x3a, x4a, x0, x1, x2, x3, x4, feature, kernels, correct, predicts, prob = unified_classifier(
        x_, x_mask_, y_, keep_prob, opt,
        class_penalty_)
    # accuracy_, loss_, train_op, W_norm_, global_step = emb_classifier(x_, x_mask_, y_, keep_prob, opt, class_penalty_)
    uidx = 0
    max_val_accuracy = np.array([0.]*10)
    max_test_accuracy = 0.

    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    np.set_printoptions(precision=3)
    np.set_printoptions(threshold=np.inf)
    saver = tf.train.Saver()

    # 日志初始化
    logging.basicConfig(
        level=logging.DEBUG,  # 定义输出到文件的log级别，大于此级别的都被输出
        format='[%(asctime)s] %(message)s',  # 定义输出log的格式
        datefmt='%Y-%m-%d %A %H:%M:%S',  # 时间
        filename="./log/unified_v13_%s_%s_%s.txt" % (opt.dataset, opt.model, opt.ngram),  # log文件名
        filemode='w')  # 写入模式“w”或“a”

    # Define a Handler and set a format which output to console
    console = logging.StreamHandler()  # 定义console handler
    console.setLevel(logging.DEBUG)  # 定义该handler级别
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    console.setFormatter(formatter)
    # Create an instance
    logging.getLogger().addHandler(console)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.GPUID)
    #config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    #config.gpu_options.allow_growth = True
    save_features = []
#    with tf.Session() as sess:#tf.Session(config=config) as sess:
        #train_writer = tf.summary.FileWriter(opt.log_path + '/train', sess.graph)
        #test_writer = tf.summary.FileWriter(opt.log_path + '/test', sess.graph)
    sess.run(tf.global_variables_initializer())
    print('Training Start! Parameters are set as:')
    logging.debug('Training Start! Parameters are set as:')
    for key in d.keys():
        print(key, ': ', d[key])
        logging.debug(key+': '+str(d[key]))
    if opt.restore:
        try:
            t_vars = tf.trainable_variables()
            save_keys = tensors_key_in_file(opt.save_path)
            ss = set([var.name for var in t_vars]) & set([s + ":0" for s in save_keys.keys()])
            cc = {var.name: var for var in t_vars}
            # only restore variables with correct shape
            ss_right_shape = set([s for s in ss if cc[s].get_shape() == save_keys[s[:-2]]])

            loader = tf.train.Saver(var_list=[var for var in t_vars if var.name in ss_right_shape])
            loader.restore(sess, opt.save_path)

            print("Loading variables from '%s'." % opt.save_path)
            print("Loaded variables:" + str(ss))

        except:
            print("No saving session, using random initialization")
            sess.run(tf.global_variables_initializer())

    try:
        iters = []
        train_loss = []
        train_acc = []
        val_acc = []
        test_acc = []
        lr_decay = 0
        for epoch in range(opt.max_epochs):
            print("********************************")
            print("Starting epoch %d" % epoch)
            logging.debug("********************************")
            logging.debug("Starting epoch %d" % epoch)
            kf = get_minibatches_idx(len(train), opt.batch_size, shuffle=True)
            for _, train_index in kf:
                uidx += 1
#                if uidx > 2:
#                    break
                sents = [train[t] for t in train_index]
                x_labels = [train_lab[t] for t in train_index]
                x_labels = np.array(x_labels)
                x_labels = x_labels.reshape((len(x_labels), opt.num_class))

                x_batch, x_batch_mask = prepare_data_for_emb(sents, opt)
                if opt.wordlabel_ratio > 0:
                    _, loss, step, W_class = sess.run([train_op, loss_, global_step, update_label],
                                                      feed_dict={x_: x_batch, x_mask_: x_batch_mask, y_: x_labels,
                                                                 keep_prob: opt.dropout,
                                                                 class_penalty_: opt.class_penalty})
                else:
                    _, loss, accuracy, step = sess.run([train_op, loss_, accuracy_,  global_step],
                                                      feed_dict={x_: x_batch, x_mask_: x_batch_mask, y_: x_labels,
                                                                 keep_prob: opt.dropout,
                                                                 class_penalty_: opt.class_penalty})



                print('iter %d loss:%f accuracy:%f' % (uidx, loss, accuracy))

                if math.isnan(loss):
                    print('xxx!')
                    break

                if uidx % opt.valid_freq == 0:#if epoch > 0 and uidx % opt.valid_freq == 0:
                    train_correct = 0.0
                    # sample evaluate accuaccy on 500 sample data
#                    kf_train = get_minibatches_idx(500, opt.batch_size, shuffle=True)
#                    for _, train_index in kf_train:
#                        train_sents = [train[t] for t in train_index]
#                        train_labels = [train_lab[t] for t in train_index]
#                        train_labels = np.array(train_labels)
#                        train_labels = train_labels.reshape((len(train_labels), opt.num_class))
#                        x_train_batch, x_train_batch_mask = prepare_data_for_emb(train_sents, opt)
#                        train_accuracy = sess.run(accuracy_,
#                                                  feed_dict={x_: x_train_batch, x_mask_: x_train_batch_mask,
#                                                             y_: train_labels, keep_prob: 1.0, class_penalty_: 0.0})
#
#                        train_correct += train_accuracy * len(train_index)
#
#                    train_accuracy = train_correct / 500
#
#                    print("-----------------------")
#                    print("Iteration %d: Training loss %f " % (uidx, loss))
#                    print("Train accuracy %f " % train_accuracy)
#                    logging.debug("-----------------------")
#                    logging.debug("Iteration %d: Training loss:%f acc: %f" % (uidx, loss, train_accuracy))
#                    iters.append(uidx)
#                    train_loss.append(loss)
#                    train_acc.append(train_accuracy)

                    val_correct = 0.0
                    kf_val = get_minibatches_idx(len(val), opt.batch_size, shuffle=True)
                    for _, val_index in kf_val:
                        val_sents = [val[t] for t in val_index]
                        val_labels = [val_lab[t] for t in val_index]
                        val_labels = np.array(val_labels)
                        val_labels = val_labels.reshape((len(val_labels), opt.num_class))
                        x_val_batch, x_val_batch_mask = prepare_data_for_emb(val_sents, opt)
                        val_accuracy = sess.run(accuracy_, feed_dict={x_: x_val_batch, x_mask_: x_val_batch_mask,
                                                                      y_: val_labels, keep_prob: 1.0,
                                                                      class_penalty_: 0.0})

                        val_correct += val_accuracy * len(val_index)

                    val_accuracy = val_correct / len(val)
                    print("Validation accuracy %f " % val_accuracy)
                    logging.debug("Validation acc: %f" % val_accuracy)
                    val_acc.append(val_accuracy)

#                    if val_accuracy > np.min(max_val_accuracy):
#                        max_val_accuracy[np.argmin(max_val_accuracy)] = val_accuracy
#
#                        test_correct = 0.0
#
#                        kf_test = get_minibatches_idx(len(test), opt.batch_size, shuffle=True)
#                        wrong_predicts = []
#                        for _, test_index in kf_test:
#                            test_sents = [test[t] for t in test_index]
#                            test_labels = [test_lab[t] for t in test_index]
#                            test_labels = np.array(test_labels)
#                            test_labels = test_labels.reshape((len(test_labels), opt.num_class))
#                            x_test_batch, x_test_batch_mask = prepare_data_for_emb(test_sents, opt)
#
#                            test_accuracy, correct_ind = sess.run([accuracy_, correct],
#                                                     feed_dict={x_: x_test_batch, x_mask_: x_test_batch_mask,
#                                                                y_: test_labels, keep_prob: 1.0,
#                                                                class_penalty_: 0.0})
#                            # wrong_num = 0
#                            for t_ind, t in enumerate(test_index):
#                                if not correct_ind[t_ind]:
#                                    wrong_predicts.append(t)
#                                    # wrong_num+=1
#
#                            test_correct += test_accuracy * len(test_index)
#                            # print(str(test_accuracy*len(test_index))+'/'+str(wrong_num)+'/'+str(len(test_index)))
#                        test_accuracy = test_correct / len(test)
#                        print("Test accuracy %f " % test_accuracy)
#                        logging.debug("Test accuracy %f" % test_accuracy)
#                        if test_accuracy > max_test_accuracy:
#                            max_test_accuracy = test_accuracy
#                            if opt.save_feature and lr_decay > 0:
#                                kf_save = get_minibatches_idx(opt.save_size, opt.batch_size)
#                                save_features = []
#                                for _, save_index in kf_save:
#                                    save_sents = [train_save[t] for t in save_index]
#                                    save_labels = [train_lab_save[t] for t in save_index]
#                                    save_labels = np.array(save_labels)
#                                    save_labels = save_labels.reshape((len(save_labels), opt.num_class))
#                                    save_batch, save_batch_mask = prepare_data_for_emb(save_sents, opt)
#
#                                    save_ops = []
#                                    if '0' in opt.model:
#                                        save_ops.append(x0a)
#                                        save_ops.append(x0v)
#                                        save_ops.append(x0)
#                                    if '1' in opt.model:
#                                        save_ops.append(x1a)
#                                        save_ops.append(x1v)
#                                        save_ops.append(x1)
#                                    if '3' in opt.model:
#                                        save_ops.append(x3a)
#                                        save_ops.append(x3)
#                                    if '4' in opt.model:
#                                        save_ops.append(x4a)
#                                        save_ops.append(x4)
#                                    save_ops.append(feature)
#                                    save_ops.append(kernels)
#                                    save_features.append(sess.run(save_ops,
#                                                             feed_dict={x_: save_batch,
#                                                                        x_mask_: save_batch_mask,
#                                                                        y_: save_labels, keep_prob: 1.0,
#                                                                        class_penalty_: 0.0}))
#                        test_acc.append(test_accuracy)
#                        np.savetxt('test_wrong_predict_iter%d.csv' % uidx, np.array(wrong_predicts),delimiter=',')
#                        if test_accuracy > decay_dict[opt.dataset] and lr_decay == 0:
#                            opt.lr = 1e-4
#                            print('Learning rate has been decayed to %f' % opt.lr)
#                            logging.debug('Learning rate has been decayed to %f' % opt.lr)
#                            lr_decay += 1
#                    else:
#                        test_acc.append(-1)
#
#            print("Epoch %d: Max Test accuracy %f" % (epoch, max_test_accuracy))
#            logging.debug("Epoch %d: Max Test accuracy %f" % (epoch, max_test_accuracy))
#            saver.save(sess, opt.save_path, global_step=epoch)
# =============================================================================
#                 保存.pb文件
# =============================================================================
#            frozen_graph = freeze_session(sess, output_names=[predicts.name])
#            tf.train.write_graph(frozen_graph, d['save_path'], "node521_dnn.pb", as_text=False)
            save_model(saver, sess, output_name=predicts.name, save_dir=d['save_path'])
            
#        print("Max Test accuracy %f " % max_test_accuracy)
#        logging.debug("Max Test accuracy %f " % max_test_accuracy)
#        if opt.save_feature:
#            save_ind = 0
#            if '0' in opt.model:
#                x0a_save = np.reshape(np.array([batch[save_ind] for batch in save_features]),
#                                       [opt.save_size, opt.maxlen * (len(opt.ngram) + 1)])
#                x0v_save = np.array(save_features[0][save_ind + 1])
#                x0_save = np.reshape(np.array([batch[save_ind + 2] for batch in save_features]),
#                                     [opt.save_size, opt.embed_size])
#                np.savetxt(
#                        "./log/unified_v13_%s_%s_%s_x0_attention.csv" % (
#                            opt.dataset, opt.model, opt.ngram),
#                        x0a_save, delimiter=',')
#                np.savetxt(
#                    "./log/unified_v13_%s_%s_%s_x0_paravector.csv" % (
#                    opt.dataset, opt.model, opt.ngram),
#                    x0v_save, delimiter=',')
#                np.save("./log/unified_v13_%s_%s_%s_x0" % (opt.dataset, opt.model, opt.ngram), x0_save)
#                save_ind += 3
#            if '1' in opt.model:
#                x1a_save = np.reshape(np.array([batch[save_ind] for batch in save_features]),
#                                       [opt.save_size, opt.maxlen * (len(opt.ngram)*2 + 2)])
#                x1v_save = np.array(save_features[0][save_ind + 1])
#                x1_save = np.reshape(np.array([batch[save_ind + 2] for batch in save_features]),
#                                     [opt.save_size, opt.embed_size])
#                np.savetxt(
#                        "./log/unified_v13_%s_%s_%s_x1_attention.csv" % (
#                            opt.dataset, opt.model, opt.ngram),
#                        x1a_save, delimiter=',')
#                np.savetxt(
#                    "./log/unified_v13_%s_%s_%s_x1_paravector.csv" % (
#                        opt.dataset, opt.model, opt.ngram),
#                    x1v_save, delimiter=',')
#                np.save("./log/unified_v13_%s_%s_%s_x1" % (opt.dataset, opt.model, opt.ngram), x1_save)
#                save_ind += 3
#            if '3' in opt.model:
#                x3a_save = [np.reshape(np.array([batch[save_ind][k_i] for batch in save_features]),
#                                       [opt.save_size, opt.maxlen]) for k_i in range(len(opt.ngram)+1)]
#                x3_save = np.reshape(np.array([batch[save_ind + 1] for batch in save_features]),
#                                     [opt.save_size, opt.embed_size])
#                for a_k in range(len(x3a_save)):
#                    np.savetxt(
#                        "./log/unified_v13_%s_%s_%s_x3_%dgram_attention.csv" % (
#                            opt.dataset, opt.model, opt.ngram, int(([1]+opt.ngram)[a_k])),
#                        x3a_save[a_k], delimiter=',')
#                np.save("./log/unified_v13_%s_%s_%s_x3" % (opt.dataset, opt.model, opt.ngram), x3_save)
#                save_ind += 2
#            if '4' in opt.model:
#                x4a_save = [np.reshape(np.array([batch[save_ind][k_i] for batch in save_features]),
#                                       [opt.save_size, opt.maxlen]) for k_i in range(len(opt.ngram)+1)]
#                x4_save = np.reshape(np.array([batch[save_ind + 1] for batch in save_features]),
#                                     [opt.save_size, opt.embed_size])
#                for a_k in range(len(x4a_save)):
#                    np.savetxt(
#                        "./log/unified_v13_%s_%s_%s_x4_%dgram_attention.csv" % (
#                            opt.dataset, opt.model, opt.ngram, int(([1]+opt.ngram)[a_k])),
#                        x4a_save[a_k], delimiter=',')
#                np.save("./log/unified_v13_%s_%s_%s_x4" % (opt.dataset, opt.model, opt.ngram), x4_save)
#                save_ind += 2
#            feature_save = np.reshape(np.array([batch[save_ind] for batch in save_features]),
#                                     [opt.save_size, opt.embed_size])
#            np.save("./log/unified_v13_%s_%s_%s_feature" % (opt.dataset, opt.model, opt.ngram), feature_save)
#            save_ind += 1
#            for k_i in range(len(opt.ngram)):
#                np.savetxt("./log/unified_v13_%s_%s_%s_%dgram_kernel.csv" % (
#                opt.dataset, opt.model, opt.ngram, opt.ngram[k_i]), save_features[0][save_ind][k_i], delimiter=',')
#        np.savetxt("./log/unified_v13_%s_%s_%s_maxacc_%f.txt" % (opt.dataset, opt.model, opt.ngram, max_test_accuracy), np.array([0]))
#        np.savetxt("./log/unified_v13_%s_%s_%s.csv" % (opt.dataset, opt.model, opt.ngram),
#                   np.array([train_loss, train_acc, val_acc, test_acc]).T, delimiter=',')

    except KeyboardInterrupt:
        np.savetxt("./log/unified_v13_%s_%s_%s.csv" % (opt.dataset, opt.model, opt.ngram),
                   np.array([train_loss, train_acc, val_acc, test_acc]).T, delimiter=',')
        print('Training interupted')
        print("Max Test accuracy %f " % max_test_accuracy)
# =============================================================================
#         threshold
# =============================================================================
    test_batch, test_batch_mask = prepare_data_for_emb(test, opt)

    [predicts_value, prob_value] = sess.run([predicts, prob], feed_dict={x_: test_batch[:39,:], x_mask_: test_batch_mask[:39,:], keep_prob: 1.0})
    l_pre = [predicts_value]
    l_prob = [prob_value]
    for i in range(40):
        print(i)
        [predicts_value, prob_value] = sess.run([predicts, prob], feed_dict={x_: test_batch[39+i*50:39+(i+1)*50,:], x_mask_: test_batch_mask[39+i*50:39+(i+1)*50,:], keep_prob: 1.0})
        l_pre.append(predicts_value)
        l_prob.append(prob_value)
    
#    print(recall_disturb(np.concatenate(l_pre).tolist(), test_lab[:,1].tolist()))
    prob_pos = np.concatenate(l_prob)[:,1].tolist()
    l_recall = []
    l_disturb = []
    for i in range(100):
        [recall, disturb] = recall_disturb(prob_pos, i / 100.0, test_lab[:,1].tolist())
        l_recall.append(recall)
        l_disturb.append(disturb)
        print(i/100.0, recall, disturb)
    roc_auc_score(test_lab[:,1],prob_pos)
