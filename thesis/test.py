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
