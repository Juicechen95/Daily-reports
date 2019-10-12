

def self_att(x, x_mask, opt, prefix=None):  # x: b * s * e
    x_mask = tf.expand_dims(x_mask, axis=-1)  # b * s * 1
    x_norm = tf.multiply(x, x_mask)  # b * s * e
    v = tf.get_variable(prefix+'Self_att_V', [opt.embed_size, 1])  # e * 1
    print('icanseeyou.', x_norm.get_shape(), v.get_shape())
    A = tf.einsum('bsl,ln->bsn', x_norm, v)  # b * s * 1
    att = partial_softmax_v2(A, x_mask, 1, prefix+'Self_att_softmax')  # b * s * 1
    x_final = tf.reduce_sum(tf.multiply(x, att), 1)
    return x_final, v, tf.squeeze(att)


def self_co_att_pooling_mask(x, x_mask, opt, name):
    x_mask = tf.expand_dims(x_mask, axis=-1)  # b * s * 1
    x_co_att = tf.multiply(x, x_mask)  # b * s * e
    if opt.att_type == 'cos':
        x_norm = tf.nn.l2_normalize(x_co_att, dim=2)  # b * s * e
        x_norm_tran = tf.transpose(x_norm, [0, 2, 1])  # b * e * s
        A = tf.einsum('bse,bek->bsk', x_norm, x_norm_tran)  # b * s * s
    elif opt.att_type == 'w':
        W = tf.get_variable(name + 'self_co_att_pooling_mask_w', [opt.embed_size, opt.embed_size])
        x_co_att_tran = tf.transpose(x_co_att, [0, 2, 1])  # b * e * s
        A = tf.einsum('bse,el,blm->bsm', x_co_att, W, x_co_att_tran)  # b * s * s
    elif opt.att_type == 'share':
        W = opt.att_w
        x_co_att_tran = tf.transpose(x_co_att, [0, 2, 1])  # b * e * s
        A = tf.einsum('bse,el,blm->bsm', x_co_att, W, x_co_att_tran)  # b * s * s
    elif opt.att_type == 'mul':
        x_tran = tf.transpose(x_co_att, [0, 2, 1])
        A = tf.einsum('bse,bek->bsk', x_co_att, x_tran)  # b * s * s
    else:
        raise ValueError("Unsupported attention method=%s" % opt.att_type)


    if opt.att_none_linear:
        A = tf.nn.tanh(A)

    if opt.att_pooling == 'mean':
        A_pool = tf.reduce_mean(A, axis=-1, keep_dims=True)  # b * s * 1
    elif opt.att_pooling == 'max':
        A_pool = tf.reduce_max(A, axis=-1, keep_dims=True)
    #elif opt.att_pooling == 'max_conv':
        #Att_v = tf.contrib.layers.conv1d(A, num_outputs=opt.kernels, kernel_size=[opt.ngram], padding='SAME',
        #                                 activation_fn=tf.nn.relu)  # b * s * s
        #A_pool = tf.reduce_max(Att_v, axis=-1, keep_dims=True)
    else:
        raise ValueError("Unsupported attention pooling method=%s" % opt.att_pooling)

    att = partial_softmax_v2(A_pool, x_mask, 1, name+'self_co_att_pooling_softmax')
    x_self_mean = tf.reduce_sum(tf.multiply(x_co_att, att), 1)  # b * e

    return x_self_mean


def self_co_att_pooling_mask_multigram(x, x_mask, opt, name):
    x_mask = tf.expand_dims(x_mask, -1)
    x_self_multigram = []
    att_scores = []
    for x_co_att in x:
        if opt.att_type == 'cos':
            x_norm = tf.nn.l2_normalize(x_co_att, dim=2)  # b * s * e
            x_norm_tran = tf.transpose(x_norm, [0, 2, 1])  # b * e * s
            A = tf.einsum('bse,bek->bsk', x_norm, x_norm_tran)  # b * s * s
        elif opt.att_type == 'w':
            W = tf.get_variable(name + 'self_co_att_pooling_mask_w', [opt.embed_size, opt.embed_size])
            x_co_att_tran = tf.transpose(x_co_att, [0, 2, 1])  # b * e * s
            A = tf.einsum('bse,el,blm->bsm', x_co_att, W, x_co_att_tran)  # b * s * s
        elif opt.att_type == 'share':
            W = opt.att_w
            x_co_att_tran = tf.transpose(x_co_att, [0, 2, 1])  # b * e * s
            A = tf.einsum('bse,el,blm->bsm', x_co_att, W, x_co_att_tran)  # b * s * s
        elif opt.att_type == 'mul':
            x_tran = tf.transpose(x_co_att, [0, 2, 1])
            A = tf.einsum('bse,bek->bsk', x_co_att, x_tran)  # b * s * s
        else:
            raise ValueError("Unsupported attention method=%s" % opt.att_type)

        if opt.att_none_linear:
            A = tf.nn.tanh(A)

        if opt.att_pooling == 'mean':
            A_pool = tf.reduce_mean(A, axis=-1, keep_dims=True)  # b * s * 1
        elif opt.att_pooling == 'max':
            A_pool = tf.reduce_max(A, axis=-1, keep_dims=True)
        #elif opt.att_pooling == 'max_conv':
        #    Att_v = tf.contrib.layers.conv1d(A, num_outputs=opt.kernels, kernel_size=[opt.ngram], padding='SAME',
        #                                     activation_fn=tf.nn.relu)  # b * s * s
        #    A_pool = tf.reduce_max(Att_v, axis=-1, keep_dims=True)
        else:
            raise ValueError("Unsupported attention pooling method=%s" % opt.att_pooling)

        att = partial_softmax_v2(A_pool, x_mask, 1, name + 'self_co_att_pooling_softmax')
        x_self_mean = tf.reduce_sum(tf.multiply(x_co_att, att), 1)  # b * e

        x_self_multigram.append(x_self_mean)
        att_scores.append(tf.squeeze(att))

    return tf.reduce_mean(tf.stack(x_self_multigram), 0), att_scores


def co_att_xy_multigram(x, x_mask, y, opt):
    x_co_att_multigram = []
    x_co_att_score = []
    for x_co_att in x:
        print('icanseeindex', x.index(x_co_att))
        if opt.co_att_type_xy == 'w':
            W = tf.get_variable("co_att_w", [opt.embed_size, opt.embed_size])
            y_tran = tf.transpose(y, [1, 0])  # e * k

            A = tf.einsum('bse,el,lk->bsk', x_co_att, W, y_tran)  # b * s * k

            att = tf.nn.softmax(A, -1)
            x_co_att = tf.einsum('bsk,ke->bse', att, y)  # b * s * e

        elif opt.co_att_type_xy == 'cos':
            x_norm = tf.nn.l2_normalize(x_co_att, dim=2)  # b * s * e
            y_norm_tran = tf.transpose(tf.nn.l2_normalize(y, dim=1), [1, 0])  # e * k

            A = tf.contrib.keras.backend.dot(x_norm, y_norm_tran)  # b * s * c

            att = tf.nn.softmax(A, -1)
            print('icanseeflag', x_co_att.get_shape(), att.get_shape(), y.get_shape())
            x_co_att = tf.einsum('bsk,ke->bse', att, y)  # b * s * e
        elif opt.co_att_type_xy == 'share':
            W = opt.att_w
            y_tran = tf.transpose(y, [1, 0])  # e * k

            A = tf.einsum('bse,el,lk->bsk', x_co_att, W, y_tran)  # b * s * k

            att = tf.nn.softmax(A, -1)
            x_co_att = tf.einsum('bsk,ke->bse', att, y)  # b * s * e
        elif opt.co_att_type_xy == 'mul':
            y_tran = tf.transpose(y, [1, 0])  # e * k

            A = tf.contrib.keras.backend.dot(x_co_att, y_tran)  # b * s * c

            att = tf.nn.softmax(A, -1)
            x_co_att = tf.einsum('bsk,ke->bse', att, y)  # b * s * e
        else:
            raise ValueError("Unsupported co-attention method=%s" % opt.co_att_type_xy)
        x_co_att_multigram.append(x_co_att)
        x_co_att_score.append(att)

    return x_co_att_multigram, x_co_att_score#tf.reduce_mean(tf.stack(x_co_att_multigram), 0)

def parse_args(argv):
	parser = ArgumentParser("DeepSensor",
					formatter_class=ArgumentDefaultsHelpFormatter,
					conflict_handler='resolve')
	## optimizer config
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
	parser.add_argument('-tfrc_path_tr', action='store', dest="tfrc_path_tr", default='/workspace/web/data/web_1_train/tfrecord', type=str,
					help="The path of the saver")
	parser.add_argument('-tfrc_path_val', action='store', dest="tfrc_path_val", default='/workspace/web/data/web_1_val/tfrecord', type=str,
					help="The path of the saver")
	parser.add_argument('-batchsize',  dest="batchsize", default=2, type=int)

	#### textnet
	parser.add_argument('-t_loadpath', action='store', dest="t_loadpath", default="./text_data/web_gambling_10.p", type=str,
					help="The path of the saver")
	parser.add_argument('-t_embpath', action='store', dest="t_embpath", default="./text_data/web_gambling_emb_10.p", type=str,
					help="The path of the saver")
	parser.add_argument('-t_train_data_path', action='store', dest='t_train_data_path', default="./text_data/web_train_data.p", type=str,
					help="The path of the saver")
	return parser.parse_args()


class Options(object):
	def __init__(self,args):
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

		d = args.__dict__
		self.lr = d['lr']
		self.dataset = d['dataset']
		self.att_type = d['intra_att']
		self.co_att_type_xy = d['co_att']
		self.att_pooling = d['pool']
		self.class_penalty = d['alpha']
		self.att_none_linear = d['att_nonlinear']
		self.model = d['model']
		#self.max_epochs = d['max_epoch']
		self.ngram = [int(i) for i in d['ngram'].split(',')]
		self.kernels = d['kernel']
		self.inter_class_penalty = d['inter_alpha']
		self.intra_class_penalty = d['intra_alpha']
		self.wordlabel_ratio = d['wordlabel_ratio']
		self.feature_aggr = d['feature_aggr']
		self.multigram_leam = d['multi_leam']
		self.leam_ngram = d['leam_ngram']
		self.ensemble = d['ensemble']
		self.save_feature = d['save_feature']
        
    opt = Options(args)
    
    x_ = tf.placeholder(tf.int32, shape=[None, opt.maxlen], name='x_')
    x_mask_ = tf.placeholder(tf.float32, shape=[None, opt.maxlen], name='x_mask_')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    y_ = tf.placeholder(tf.float32, shape=[None, opt.num_class], name='y_')
    class_penalty_ = tf.placeholder(tf.float32, shape=(), name='class_penalty_')
    accuracy_, loss_, train_op, W_norm_, global_step, update_label, x0v, x0a, xya, x1v, x1a, x2a, x3a, x4a, x0, x1, x2, x3, x4, feature, kernels, correct, predicts, prob = unified_classifier(
        x_, x_mask_, y_, keep_prob, opt,
        class_penalty_)

import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
import math


def embedding(features, opt, prefix='', is_reuse=None):
    """Customized function to transform batched x into embeddings."""
    # Convert indexes of words into embeddings.
    with tf.variable_scope(prefix + 'embed', reuse=is_reuse):
        if opt.fix_emb:
            assert (hasattr(opt, 'W_emb'))
            assert (np.shape(np.array(opt.W_emb)) == (opt.n_words, opt.embed_size))
            W = tf.get_variable('W', initializer=opt.W_emb, trainable=True)
            print("initialize word embedding finished")
        else:
            weightInit = tf.random_uniform_initializer(-0.001, 0.001)
            W = tf.get_variable('W', [opt.n_words, opt.embed_size], initializer=weightInit)
    if hasattr(opt, 'relu_w') and opt.relu_w:
        W = tf.nn.relu(W)

    word_vectors = tf.nn.embedding_lookup(W, features)
    print('initialize word embedding finishedssss', features.get_shape(), word_vectors.get_shape())
    return word_vectors, W


def embedding_wordlabel(features, opt, wordlabel_idx, prefix='', is_reuse=None):
    """Customized function to transform batched x into embeddings."""
    # Convert indexes of words into embeddings.
    with tf.variable_scope(prefix + 'embed', reuse=is_reuse):
        if opt.fix_emb:
            assert (hasattr(opt, 'W_emb'))
            assert (np.shape(np.array(opt.W_emb)) == (opt.n_words, opt.embed_size))
            W = tf.get_variable('W', initializer=opt.W_emb, trainable=False)
            print("initialize word embedding finished")
        else:
            weightInit = tf.random_uniform_initializer(-0.001, 0.001)
            W = tf.get_variable('W', [opt.n_words, opt.embed_size], initializer=weightInit)
    if hasattr(opt, 'relu_w') and opt.relu_w:
        W = tf.nn.relu(W)

    word_vectors = tf.nn.embedding_lookup(W, features)

    word_label_list = []
    for yi_list in wordlabel_idx:
        word_label_list.append(tf.reduce_mean(tf.nn.embedding_lookup(W, yi_list), axis=0))
    word_label = tf.stack(word_label_list)
    return word_vectors, W, word_label


def embedding_class(features, opt, prefix='', is_reuse=None):
    """Customized function to transform batched y into embeddings."""
    # Convert indexes of words into embeddings.
    with tf.variable_scope(prefix + 'embed', reuse=is_reuse):
        if opt.fix_emb:
            assert (hasattr(opt, 'W_class_emb'))
            W = tf.get_variable('W_class', initializer=opt.W_class_emb, trainable=True)
            print("initialize class embedding finished")
        else:
            weightInit = tf.random_uniform_initializer(-0.001, 0.001)
            W = tf.get_variable('W_class', [opt.num_class, opt.embed_size], initializer=weightInit)
    if hasattr(opt, 'relu_w') and opt.relu_w:
        W = tf.nn.relu(W)
    word_vectors = tf.nn.embedding_lookup(W, features)

    return word_vectors, W


def update_label_embedding(W_class, prefix=''):
    with tf.variable_scope(prefix + 'embed', reuse=True):
        W = tf.get_variable('W_class')
        tf.assign(W, W_class)
    return W


def embedding_class_v2(features, opt, prefix='', is_reuse=None):
    """Customized function to transform batched y into embeddings."""
    # Convert indexes of words into embeddings.
    with tf.variable_scope(prefix + 'embed', reuse=is_reuse):
        if opt.fix_emb:
            assert (hasattr(opt, 'W_class_emb'))
            W = tf.get_variable('W_class', initializer=opt.W_class_emb, trainable=False)
            print("initialize class embedding finished")
        else:
            weightInit = tf.random_uniform_initializer(-0.001, 0.001)
            W = tf.get_variable('W_class', [opt.num_class, opt.embed_size], initializer=weightInit)
    if hasattr(opt, 'relu_w') and opt.relu_w:
        W = tf.nn.relu(W)
    word_vectors = tf.nn.embedding_lookup(W, features)

    return word_vectors, W


def embedding_class_wordlabel(features, opt, wordemb, prefix='', is_reuse=None):
    """Customized function to transform batched y into embeddings."""
    # Convert indexes of words into embeddings.
    with tf.variable_scope(prefix + 'word_label_embed', reuse=is_reuse):
        assert (hasattr(opt, 'word_label'))
        W = tf.get_variable('Word_label', initializer=opt.wordlabel_embedding, trainable=True)
        print("initialize word label embedding finished")
        word_vectors = tf.nn.embedding_lookup(W, features)

    return word_vectors, W


def att_emb_ngram_encoder_maxout(x_emb, x_mask, W_class, W_class_tran, opt):
    x_mask = tf.expand_dims(x_mask, axis=-1)  # b * s * 1
    x_emb_0 = tf.squeeze(x_emb, )  # b * s * e
    print('greedisgood', x_emb_0.get_shape())
    x_emb_1 = tf.multiply(x_emb_0, x_mask)  # b * s * e

    x_emb_norm = tf.nn.l2_normalize(x_emb_1, dim=2)  # b * s * e
    W_class_norm = tf.nn.l2_normalize(W_class_tran, dim=0)  # e * c
    G = tf.contrib.keras.backend.dot(x_emb_norm, W_class_norm)  # b * s * c
    x_full_emb = x_emb_0
    Att_v = tf.contrib.layers.conv1d(G, num_outputs=opt.kernels, kernel_size=[opt.leam_ngram], padding='SAME',
                                     activation_fn=tf.nn.relu)  # b * s *  c

    Att_v_m = tf.reduce_max(Att_v, axis=-1, keep_dims=True)
    Att_v_max = partial_softmax(Att_v_m, x_mask, 1, 'Att_v_max')  # b * s * 1

    x_att = tf.multiply(x_full_emb, Att_v_max)
    H_enc = tf.reduce_sum(x_att, axis=1)
    return H_enc, tf.squeeze(Att_v_max)


def att_emb_ngram_encoder_maxout_multigram(x_multigram, x_mask, W_class, W_class_tran, opt):
    x_enc_multigram = []
    att_scores = []
    for x_emb in x_multigram:
        x_emb_norm = tf.nn.l2_normalize(x_emb, dim=2)  # b * s * e
        W_class_norm = tf.nn.l2_normalize(W_class_tran, dim=0)  # e * c
        G = tf.contrib.keras.backend.dot(x_emb_norm, W_class_norm)  # b * s * c

        Att_v_m = tf.reduce_max(G, axis=-1, keep_dims=True)
        Att_v_max = partial_softmax(Att_v_m, x_mask, 1, 'Att_v_max')  # b * s * 1

        x_att = tf.multiply(x_emb, Att_v_max)
        H_enc = tf.reduce_sum(x_att, axis=1)
        x_enc_multigram.append(H_enc)
        att_scores.append(tf.squeeze(Att_v_max))

    return tf.reduce_mean(tf.stack(x_enc_multigram), 0), att_scores



def att_emb_ngram_encoder_cnn(x_emb, x_mask, W_class, W_class_tran, opt):
    x_mask = tf.expand_dims(x_mask, axis=-1)  # b * s * 1
    x_emb_0 = tf.squeeze(x_emb, )  # b * s * e
    x_emb_1 = tf.multiply(x_emb_0, x_mask)  # b * s * e

    H = tf.contrib.layers.conv2d(x_emb_0, num_outputs=opt.embed_size, kernel_size=[10], padding='SAME',
                                 activation_fn=tf.nn.relu)  # b * s *  c

    G = tf.contrib.keras.backend.dot(H, W_class_tran)  # b * s * c
    Att_v_max = partial_softmax(G, x_mask, 1, 'Att_v_max')  # b * s * c

    x_att = tf.contrib.keras.backend.batch_dot(tf.transpose(H, [0, 2, 1]), Att_v_max)
    H_enc = tf.squeeze(x_att)
    return H_enc


def aver_emb_encoder(x_emb, x_mask):
    """ compute the average over all word embeddings """
    x_mask = tf.expand_dims(x_mask, axis=-1)
    # x_mask = tf.expand_dims(x_mask, axis=-1)  # batch L 1 1

    x_sum = tf.multiply(x_emb, x_mask)  # batch L emb
    H_enc_0 = tf.reduce_sum(x_sum, axis=1, keep_dims=True)  # batch 1 emb
    H_enc = tf.squeeze(H_enc_0, [1, ])  # batch emb
    x_mask_sum = tf.reduce_sum(x_mask, axis=1, keep_dims=True)  # batch 1 1
    x_mask_sum = tf.squeeze(x_mask_sum, [2, ])  # batch 1

    # pdb.set_trace()

    H_enc = H_enc / x_mask_sum  # batch emb

    return H_enc


def gru_encoder(X_emb, opt, prefix='', is_reuse=None, res=None):
    with tf.variable_scope(prefix + 'gru_encoder', reuse=True):
        cell_fw = tf.contrib.rnn.GRUCell(opt.n_hid)
        cell_bw = tf.contrib.rnn.GRUCell(opt.n_hid)
    with tf.variable_scope(prefix + 'gru_encoder', reuse=is_reuse):
        weightInit = tf.random_uniform_initializer(-0.001, 0.001)

        packed_output, state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, X_emb, dtype=tf.float32)
        h_fw = state[0]
        h_bw = state[1]

        hidden = tf.concat((h_fw, h_bw), 1)

        hidden = tf.nn.l2_normalize(hidden, 1)
    return hidden, res


def discriminator_1layer(H, opt, dropout, prefix='', num_outputs=1, is_reuse=None):
    # last layer must be linear
    H = tf.squeeze(H)
    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    H_dis = layers.fully_connected(tf.nn.dropout(H, keep_prob=dropout), num_outputs=opt.H_dis,
                                   biases_initializer=biasInit, activation_fn=tf.nn.relu, scope=prefix + 'dis_1',
                                   reuse=is_reuse)
    return H_dis


def discriminator_0layer(H, opt, dropout, prefix='', num_outputs=1, is_reuse=None):
    H = tf.squeeze(H)
    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    logits = layers.linear(tf.nn.dropout(H, keep_prob=dropout), num_outputs=num_outputs, biases_initializer=biasInit,
                           scope=prefix + 'dis', reuse=is_reuse)
    return logits


def linear_layer(x, output_dim, prefix):
    input_dim = x.get_shape().as_list()[1]
    thres = np.sqrt(6.0 / (input_dim + output_dim))
    W = tf.get_variable("W", [input_dim, output_dim], scope=prefix + '_W',
                        initializer=tf.random_uniform_initializer(minval=-thres, maxval=thres))
    b = tf.get_variable("b", [output_dim], scope=prefix + '_b', initializer=tf.constant_initializer(0.0))
    return tf.matmul(x, W) + b


def discriminator_2layer(H, opt, dropout, prefix='', num_outputs=1, is_reuse=None):
    # last layer must be linear
    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    H_dis = layers.fully_connected(tf.nn.dropout(H, keep_prob=dropout), num_outputs=opt.H_dis,
                                   biases_initializer=biasInit, activation_fn=tf.nn.relu, scope=prefix + 'dis_1',
                                   reuse=is_reuse)
    logits = layers.linear(tf.nn.dropout(H_dis, keep_prob=dropout), num_outputs=num_outputs,
                           biases_initializer=biasInit, scope=prefix + 'dis_2', reuse=is_reuse)
    return logits


def discriminator_3layer(H, opt, dropout, prefix='', num_outputs=1, is_reuse=None):
    # last layer must be linear
    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    H_dis = layers.fully_connected(tf.nn.dropout(H, keep_prob=dropout), num_outputs=opt.H_dis,
                                   biases_initializer=biasInit, activation_fn=tf.nn.relu, scope=prefix + 'dis_1',
                                   reuse=is_reuse)
    H_dis = layers.fully_connected(tf.nn.dropout(H_dis, keep_prob=dropout), num_outputs=opt.H_dis,
                                   biases_initializer=biasInit, activation_fn=tf.nn.relu, scope=prefix + 'dis_2',
                                   reuse=is_reuse)
    logits = layers.linear(tf.nn.dropout(H_dis, keep_prob=dropout), num_outputs=num_outputs,
                           biases_initializer=biasInit, scope=prefix + 'dis_3', reuse=is_reuse)
    return logits


def partial_softmax(logits, weights, dim, name, ):
    with tf.name_scope('partial_softmax'):
        print('logits', logits)
        exp_logits = tf.exp(logits)
        if len(exp_logits.get_shape()) == len(weights.get_shape()):
            exp_logits_weighted = tf.multiply(exp_logits, weights)
        else:
            exp_logits_weighted = tf.multiply(exp_logits, tf.expand_dims(weights, -1))
        exp_logits_sum = tf.reduce_sum(exp_logits_weighted, axis=dim, keep_dims=True)
        partial_softmax_score = tf.div(exp_logits_weighted, exp_logits_sum, name=name)
        return partial_softmax_score


def partial_softmax_v2(logits, weights, dim, name, ):
    with tf.name_scope('partial_softmax'):
        mask = tf.multiply(tf.ones_like(logits) * (-65530), (1 - weights))
        logits_mask = tf.add(tf.multiply(logits, weights), mask)
        partial_softmax_score = tf.multiply(tf.nn.softmax(logits_mask, dim), weights)
        return partial_softmax_score


def partial_softmax_v3(logits, weights, dim, name, ):
    with tf.name_scope('partial_softmax'):
        logits_max = tf.reduce_max(logits, axis=1, keep_dims=True)
        logits_sub = tf.expand_dims(tf.subtract(logits, logits_max), -1)
        exp_logits = tf.exp(logits_sub)
        exp_logits_weighted = tf.multiply(exp_logits, weights)
        exp_logits_sum = tf.reduce_sum(exp_logits_weighted, axis=dim, keep_dims=True)
        partial_softmax_score = tf.div(logits_sub, exp_logits_sum)
        return partial_softmax_score
   
import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
import math


def embedding(features, opt, prefix='', is_reuse=None):
    """Customized function to transform batched x into embeddings."""
    # Convert indexes of words into embeddings.
    with tf.variable_scope(prefix + 'embed', reuse=is_reuse):
        if opt.fix_emb:
            assert (hasattr(opt, 'W_emb'))
            assert (np.shape(np.array(opt.W_emb)) == (opt.n_words, opt.embed_size))
            W = tf.get_variable('W', initializer=opt.W_emb, trainable=True)
            print("initialize word embedding finished")
        else:
            weightInit = tf.random_uniform_initializer(-0.001, 0.001)
            W = tf.get_variable('W', [opt.n_words, opt.embed_size], initializer=weightInit)
    if hasattr(opt, 'relu_w') and opt.relu_w:
        W = tf.nn.relu(W)

    word_vectors = tf.nn.embedding_lookup(W, features)
    print('initialize word embedding finishedssss', features.get_shape(), word_vectors.get_shape())
    return word_vectors, W


def embedding_wordlabel(features, opt, wordlabel_idx, prefix='', is_reuse=None):
    """Customized function to transform batched x into embeddings."""
    # Convert indexes of words into embeddings.
    with tf.variable_scope(prefix + 'embed', reuse=is_reuse):
        if opt.fix_emb:
            assert (hasattr(opt, 'W_emb'))
            assert (np.shape(np.array(opt.W_emb)) == (opt.n_words, opt.embed_size))
            W = tf.get_variable('W', initializer=opt.W_emb, trainable=False)
            print("initialize word embedding finished")
        else:
            weightInit = tf.random_uniform_initializer(-0.001, 0.001)
            W = tf.get_variable('W', [opt.n_words, opt.embed_size], initializer=weightInit)
    if hasattr(opt, 'relu_w') and opt.relu_w:
        W = tf.nn.relu(W)

    word_vectors = tf.nn.embedding_lookup(W, features)

    word_label_list = []
    for yi_list in wordlabel_idx:
        word_label_list.append(tf.reduce_mean(tf.nn.embedding_lookup(W, yi_list), axis=0))
    word_label = tf.stack(word_label_list)
    return word_vectors, W, word_label


def embedding_class(features, opt, prefix='', is_reuse=None):
    """Customized function to transform batched y into embeddings."""
    # Convert indexes of words into embeddings.
    with tf.variable_scope(prefix + 'embed', reuse=is_reuse):
        if opt.fix_emb:
            assert (hasattr(opt, 'W_class_emb'))
            W = tf.get_variable('W_class', initializer=opt.W_class_emb, trainable=True)
            print("initialize class embedding finished")
        else:
            weightInit = tf.random_uniform_initializer(-0.001, 0.001)
            W = tf.get_variable('W_class', [opt.num_class, opt.embed_size], initializer=weightInit)
    if hasattr(opt, 'relu_w') and opt.relu_w:
        W = tf.nn.relu(W)
    word_vectors = tf.nn.embedding_lookup(W, features)

    return word_vectors, W


def update_label_embedding(W_class, prefix=''):
    with tf.variable_scope(prefix + 'embed', reuse=True):
        W = tf.get_variable('W_class')
        tf.assign(W, W_class)
    return W


def embedding_class_v2(features, opt, prefix='', is_reuse=None):
    """Customized function to transform batched y into embeddings."""
    # Convert indexes of words into embeddings.
    with tf.variable_scope(prefix + 'embed', reuse=is_reuse):
        if opt.fix_emb:
            assert (hasattr(opt, 'W_class_emb'))
            W = tf.get_variable('W_class', initializer=opt.W_class_emb, trainable=False)
            print("initialize class embedding finished")
        else:
            weightInit = tf.random_uniform_initializer(-0.001, 0.001)
            W = tf.get_variable('W_class', [opt.num_class, opt.embed_size], initializer=weightInit)
    if hasattr(opt, 'relu_w') and opt.relu_w:
        W = tf.nn.relu(W)
    word_vectors = tf.nn.embedding_lookup(W, features)

    return word_vectors, W


def embedding_class_wordlabel(features, opt, wordemb, prefix='', is_reuse=None):
    """Customized function to transform batched y into embeddings."""
    # Convert indexes of words into embeddings.
    with tf.variable_scope(prefix + 'word_label_embed', reuse=is_reuse):
        assert (hasattr(opt, 'word_label'))
        W = tf.get_variable('Word_label', initializer=opt.wordlabel_embedding, trainable=True)
        print("initialize word label embedding finished")
        word_vectors = tf.nn.embedding_lookup(W, features)

    return word_vectors, W


def att_emb_ngram_encoder_maxout(x_emb, x_mask, W_class, W_class_tran, opt):
    x_mask = tf.expand_dims(x_mask, axis=-1)  # b * s * 1
    x_emb_0 = tf.squeeze(x_emb, )  # b * s * e
    print('greedisgood', x_emb_0.get_shape())
    x_emb_1 = tf.multiply(x_emb_0, x_mask)  # b * s * e

    x_emb_norm = tf.nn.l2_normalize(x_emb_1, dim=2)  # b * s * e
    W_class_norm = tf.nn.l2_normalize(W_class_tran, dim=0)  # e * c
    G = tf.contrib.keras.backend.dot(x_emb_norm, W_class_norm)  # b * s * c
    x_full_emb = x_emb_0
    Att_v = tf.contrib.layers.conv1d(G, num_outputs=opt.kernels, kernel_size=[opt.leam_ngram], padding='SAME',
                                     activation_fn=tf.nn.relu)  # b * s *  c

    Att_v_m = tf.reduce_max(Att_v, axis=-1, keep_dims=True)
    Att_v_max = partial_softmax(Att_v_m, x_mask, 1, 'Att_v_max')  # b * s * 1

    x_att = tf.multiply(x_full_emb, Att_v_max)
    H_enc = tf.reduce_sum(x_att, axis=1)
    return H_enc, tf.squeeze(Att_v_max)


def att_emb_ngram_encoder_maxout_multigram(x_multigram, x_mask, W_class, W_class_tran, opt):
    x_enc_multigram = []
    att_scores = []
    for x_emb in x_multigram:
        x_emb_norm = tf.nn.l2_normalize(x_emb, dim=2)  # b * s * e
        W_class_norm = tf.nn.l2_normalize(W_class_tran, dim=0)  # e * c
        G = tf.contrib.keras.backend.dot(x_emb_norm, W_class_norm)  # b * s * c

        Att_v_m = tf.reduce_max(G, axis=-1, keep_dims=True)
        Att_v_max = partial_softmax(Att_v_m, x_mask, 1, 'Att_v_max')  # b * s * 1

        x_att = tf.multiply(x_emb, Att_v_max)
        H_enc = tf.reduce_sum(x_att, axis=1)
        x_enc_multigram.append(H_enc)
        att_scores.append(tf.squeeze(Att_v_max))

    return tf.reduce_mean(tf.stack(x_enc_multigram), 0), att_scores



def att_emb_ngram_encoder_cnn(x_emb, x_mask, W_class, W_class_tran, opt):
    x_mask = tf.expand_dims(x_mask, axis=-1)  # b * s * 1
    x_emb_0 = tf.squeeze(x_emb, )  # b * s * e
    x_emb_1 = tf.multiply(x_emb_0, x_mask)  # b * s * e

    H = tf.contrib.layers.conv2d(x_emb_0, num_outputs=opt.embed_size, kernel_size=[10], padding='SAME',
                                 activation_fn=tf.nn.relu)  # b * s *  c

    G = tf.contrib.keras.backend.dot(H, W_class_tran)  # b * s * c
    Att_v_max = partial_softmax(G, x_mask, 1, 'Att_v_max')  # b * s * c

    x_att = tf.contrib.keras.backend.batch_dot(tf.transpose(H, [0, 2, 1]), Att_v_max)
    H_enc = tf.squeeze(x_att)
    return H_enc


def aver_emb_encoder(x_emb, x_mask):
    """ compute the average over all word embeddings """
    x_mask = tf.expand_dims(x_mask, axis=-1)
    # x_mask = tf.expand_dims(x_mask, axis=-1)  # batch L 1 1

    x_sum = tf.multiply(x_emb, x_mask)  # batch L emb
    H_enc_0 = tf.reduce_sum(x_sum, axis=1, keep_dims=True)  # batch 1 emb
    H_enc = tf.squeeze(H_enc_0, [1, ])  # batch emb
    x_mask_sum = tf.reduce_sum(x_mask, axis=1, keep_dims=True)  # batch 1 1
    x_mask_sum = tf.squeeze(x_mask_sum, [2, ])  # batch 1

    # pdb.set_trace()

    H_enc = H_enc / x_mask_sum  # batch emb

    return H_enc


def gru_encoder(X_emb, opt, prefix='', is_reuse=None, res=None):
    with tf.variable_scope(prefix + 'gru_encoder', reuse=True):
        cell_fw = tf.contrib.rnn.GRUCell(opt.n_hid)
        cell_bw = tf.contrib.rnn.GRUCell(opt.n_hid)
    with tf.variable_scope(prefix + 'gru_encoder', reuse=is_reuse):
        weightInit = tf.random_uniform_initializer(-0.001, 0.001)

        packed_output, state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, X_emb, dtype=tf.float32)
        h_fw = state[0]
        h_bw = state[1]

        hidden = tf.concat((h_fw, h_bw), 1)

        hidden = tf.nn.l2_normalize(hidden, 1)
    return hidden, res


def discriminator_1layer(H, opt, dropout, prefix='', num_outputs=1, is_reuse=None):
    # last layer must be linear
    H = tf.squeeze(H)
    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    H_dis = layers.fully_connected(tf.nn.dropout(H, keep_prob=dropout), num_outputs=opt.H_dis,
                                   biases_initializer=biasInit, activation_fn=tf.nn.relu, scope=prefix + 'dis_1',
                                   reuse=is_reuse)
    return H_dis


def discriminator_0layer(H, opt, dropout, prefix='', num_outputs=1, is_reuse=None):
    H = tf.squeeze(H)
    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    logits = layers.linear(tf.nn.dropout(H, keep_prob=dropout), num_outputs=num_outputs, biases_initializer=biasInit,
                           scope=prefix + 'dis', reuse=is_reuse)
    return logits


def linear_layer(x, output_dim, prefix):
    input_dim = x.get_shape().as_list()[1]
    thres = np.sqrt(6.0 / (input_dim + output_dim))
    W = tf.get_variable("W", [input_dim, output_dim], scope=prefix + '_W',
                        initializer=tf.random_uniform_initializer(minval=-thres, maxval=thres))
    b = tf.get_variable("b", [output_dim], scope=prefix + '_b', initializer=tf.constant_initializer(0.0))
    return tf.matmul(x, W) + b


def discriminator_2layer(H, opt, dropout, prefix='', num_outputs=1, is_reuse=None):
    # last layer must be linear
    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    H_dis = layers.fully_connected(tf.nn.dropout(H, keep_prob=dropout), num_outputs=opt.H_dis,
                                   biases_initializer=biasInit, activation_fn=tf.nn.relu, scope=prefix + 'dis_1',
                                   reuse=is_reuse)
    logits = layers.linear(tf.nn.dropout(H_dis, keep_prob=dropout), num_outputs=num_outputs,
                           biases_initializer=biasInit, scope=prefix + 'dis_2', reuse=is_reuse)
    return logits


def discriminator_3layer(H, opt, dropout, prefix='', num_outputs=1, is_reuse=None):
    # last layer must be linear
    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    H_dis = layers.fully_connected(tf.nn.dropout(H, keep_prob=dropout), num_outputs=opt.H_dis,
                                   biases_initializer=biasInit, activation_fn=tf.nn.relu, scope=prefix + 'dis_1',
                                   reuse=is_reuse)
    H_dis = layers.fully_connected(tf.nn.dropout(H_dis, keep_prob=dropout), num_outputs=opt.H_dis,
                                   biases_initializer=biasInit, activation_fn=tf.nn.relu, scope=prefix + 'dis_2',
                                   reuse=is_reuse)
    logits = layers.linear(tf.nn.dropout(H_dis, keep_prob=dropout), num_outputs=num_outputs,
                           biases_initializer=biasInit, scope=prefix + 'dis_3', reuse=is_reuse)
    return logits


def partial_softmax(logits, weights, dim, name, ):
    with tf.name_scope('partial_softmax'):
        print('logits', logits)
        exp_logits = tf.exp(logits)
        if len(exp_logits.get_shape()) == len(weights.get_shape()):
            exp_logits_weighted = tf.multiply(exp_logits, weights)
        else:
            exp_logits_weighted = tf.multiply(exp_logits, tf.expand_dims(weights, -1))
        exp_logits_sum = tf.reduce_sum(exp_logits_weighted, axis=dim, keep_dims=True)
        partial_softmax_score = tf.div(exp_logits_weighted, exp_logits_sum, name=name)
        return partial_softmax_score


def partial_softmax_v2(logits, weights, dim, name, ):
    with tf.name_scope('partial_softmax'):
        mask = tf.multiply(tf.ones_like(logits) * (-65530), (1 - weights))
        logits_mask = tf.add(tf.multiply(logits, weights), mask)
        partial_softmax_score = tf.multiply(tf.nn.softmax(logits_mask, dim), weights)
        return partial_softmax_score


def partial_softmax_v3(logits, weights, dim, name, ):
    with tf.name_scope('partial_softmax'):
        logits_max = tf.reduce_max(logits, axis=1, keep_dims=True)
        logits_sub = tf.expand_dims(tf.subtract(logits, logits_max), -1)
        exp_logits = tf.exp(logits_sub)
        exp_logits_weighted = tf.multiply(exp_logits, weights)
        exp_logits_sum = tf.reduce_sum(exp_logits_weighted, axis=dim, keep_dims=True)
        partial_softmax_score = tf.div(logits_sub, exp_logits_sum)
        return partial_softmax_score

    
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
