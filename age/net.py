
import numpy as np
import tensorflow as tf
import os

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class Network(object):
    def __init__(
            self,
            n_output=2,
            n_length=128,
            learning_rate=0.01,
            batch_size=32,
            channel = 1,
            output_graph=False,
            use_ckpt = True
    ):


        self.n_length = n_length # width or height of input matrix
        self.lr = learning_rate
        self.batch_size = batch_size
        self.channel = channel # num of channel
        self.learn_step_counter = 0
        self.global_step = tf.Variable(tf.constant(1))
        self.global_counter = 1 # equal to self.global_step
        self.n_output = n_output
        self._build_net()

        # e_params = tf.get_collection('eval_net_params')

        # self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        # self.replace_eval_op = [tf.assign(e, t) for e, t in zip(e_params,t_params)]
        # assign e to t

        self.sess = tf.Session()
        self.saver = tf.train.Saver(tf.global_variables())
        # self.dir_path = '/home/hengtong/project/protein/ProteinHP_DQN/heng_model/save_weight/'
        self.dir_path = os.path.dirname(os.path.realpath(__file__))

        if output_graph:
            # tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        if use_ckpt:
            self.restore_parameters()
        else:
            self.sess.run(tf.global_variables_initializer()) # train step

        self.cost_his = []


    def conv2d(self,x, W, stride,pad):
        # stride [1, x_movement, y_movement, 1]
        # Must have strides[0] = strides[3] = 1
        return tf.nn.conv2d(x, W, strides = stride, padding=pad)

    def max_pool(self,x,k,stride,pad):
        # stride [1, x_movement, y_movement, 1]
        return tf.nn.max_pool(x, ksize=k, strides=stride, padding=pad)

    def BN_fc(self,x,dim):
        # x is input size,dim is batch size
        mean_value,var_value = tf.nn.moments(x,[0])
        scales = tf.Variable(tf.ones([dim]))
        betas = tf.Variable(tf.zeros([dim]))
        epsilon = 1e-3
        return tf.nn.batch_normalization(x,mean_value,var_value,scales,betas,epsilon)


    def _build_net(self):
        # ------------------ build Gender_Net ------------------
        self.xs = tf.placeholder(tf.float32, [None, self.n_length, self.n_length, self.channel], name='s')  # input
        # self.s = tf.reshape(self.xs, [-1, self.n_features, self.n_features, self.channel])
        # self.keep_prob = tf.placeholder(tf.float32)
        self.labels = tf.placeholder(tf.float32,[None,self.n_output],name='age_labels')

        with tf.variable_scope('gender_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, w_initializer, b_initializer = \
                ['gender_net_params', tf.GraphKeys.GLOBAL_VARIABLES], \
                tf.truncated_normal_initializer(0., 0.1), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('convol1'):
                w1_conv = tf.get_variable('w1_conv', [7,7,self.channel,32], initializer=w_initializer, collections=c_names)
                b1_conv = tf.get_variable('b1_conv', [1,32], initializer=b_initializer, collections=c_names)
                h_conv1 = tf.nn.relu(self.conv2d(self.xs, w1_conv,stride=[1,2,2,1],pad='SAME') + b1_conv)  # output size 64*64*32
                lrn1 = tf.nn.local_response_normalization(h_conv1, alpha=0.0001, beta=0.75)
                h_pool1 = self.max_pool(lrn1,k=[1,3,3,1],stride=[1,2,2,1],pad='SAME') # output size 32*32*32

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('convol2'):
                w2_conv = tf.get_variable('w2_conv', [5,5,32,64], initializer=w_initializer, collections=c_names)
                b2_conv = tf.get_variable('b2_conv', [1,64], initializer=b_initializer, collections=c_names)
                h_conv2 = tf.nn.relu(self.conv2d(h_pool1, w2_conv,stride=[1,1,1,1],pad='SAME') + b2_conv)  # output size 32*32*64
                lrn2 = tf.nn.local_response_normalization(h_conv2, alpha=0.0001, beta=0.75)
                h_pool2 = self.max_pool(lrn2,k=[1,3,3,1],stride=[1,2,2,1],pad='SAME') # output 16*16*64

            with tf.variable_scope('convol3'):
                w3_conv = tf.get_variable('w2_conv', [3,3,64,64], initializer=w_initializer, collections=c_names)
                b3_conv = tf.get_variable('b2_conv', [1,64], initializer=b_initializer, collections=c_names)
                h_conv3 = tf.nn.relu(self.conv2d(h_pool2, w3_conv,stride=[1,1,1,1],pad='SAME') + b3_conv)  # output size 16*16*64
                # lrn3 = tf.nn.local_response_normalization(h_conv3, alpha=0.0001, beta=0.75)
                h_pool3 = self.max_pool(h_conv3,k=[1,3,3,1],stride=[1,2,2,1],pad='SAME') # output 8*8*64

            # fully connected layer 1
            with tf.variable_scope('fyl1'):
                w1_fu = tf.get_variable('w1_fu',[8*8*64,512],initializer=w_initializer, collections=c_names)
                b1_fu = tf.get_variable('b1_fu',[1,512],initializer=b_initializer, collections=c_names)
                h_pool3_flat = tf.reshape(h_pool3, [-1, 8*8*64])
                bn_in_fc1 = tf.matmul(h_pool3_flat, w1_fu) + b1_fu
                # bn_out_fc1 = self.BN_fc(bn_in_fc1,512)
                h_fc1 = tf.nn.relu(bn_in_fc1)
                # h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

            # fully connected layer 2,gender
            with tf.variable_scope('fyl2_pro'):
                w2_fu = tf.get_variable('w2_fu_pro', [512,self.n_output], initializer=w_initializer, collections=c_names)
                b2_fu = tf.get_variable('b2_fu_pro', [1,self.n_output], initializer=b_initializer, collections=c_names)
                pro = tf.matmul(h_fc1, w2_fu) + b2_fu
                # self.q_eval = self.BN_fc(bn_in_fc2, self.n_actions)
                self.val_output = tf.nn.softmax(pro)


        with tf.variable_scope('loss'):

            # corss entropy
            cross_entropy = -tf.reduce_mean(self.labels*tf.log(self.val_output))
            # self.loss = softmax_cross_entropy_with_logits(self.labels,self.val_output)
            # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.val_output))


            # L2 regularization for the fully connected parameters.
            regularizers = (
                            tf.nn.l2_loss(w2_fu) + tf.nn.l2_loss(b2_fu) +
                            tf.nn.l2_loss(w1_fu) + tf.nn.l2_loss(b1_fu) +
                            tf.nn.l2_loss(w3_conv) + tf.nn.l2_loss(b3_conv) +
                            tf.nn.l2_loss(w2_conv) + tf.nn.l2_loss(b2_conv) +
                            tf.nn.l2_loss(w1_conv) + tf.nn.l2_loss(b1_conv)
                            )
            self.loss = cross_entropy + 5e-4 *regularizers

        with tf.variable_scope('train'):

            learning_rate = tf.train.exponential_decay(
                self.lr,  # Base learning rate.
                #batch * self.batch_size,  # Current index into the dataset.
                self.global_step,
                5000,  # Decay step.
                0.98,  # Decay rate.
                staircase=True)
            self._train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(self.loss, global_step=self.global_step)

            # self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss) # normal training
            # self._train_op = tf.train.MomentumOptimizer(self.lr,0.9).minimize(self.loss)
            # learning_rate = tf.train.exponential_decay(learning_rate=self.lr, global_step=self.global_step,
            #                                            decay_steps=10000, decay_rate=0.96, staircase=True)
            # grad_norm = 8
            # tvars = tf.trainable_variables()
            # grads,_ = tf.clip_by_global_norm(tf.gradients(self.loss,tvars),grad_norm) # adding clipping
            # opt = tf.train.RMSPropOptimizer(self.lr)
            # self._train_op = opt.apply_gradients(zip(grads,tvars))

    def accuracy(self,predictions, labels):
        """
        Get accuracy
        :param predictions:
        :param labels:
        :return: accuracy
        """
        size = labels.shape[0]
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
                / size)

    def get_accuracy_rate(self,x,y):
        cost,predictions = self.sess.run([self.loss,self.val_output],
                                                  feed_dict={
                                                      self.labels: y,
                                                      self.xs: x
                                                  })
        accu_rate = self.accuracy(predictions, y)
        return cost,accu_rate

    def learn(self,x,y):

        # train eval network
        _, self.cost= self.sess.run([self._train_op, self.loss],
                                     feed_dict={
                                                self.labels: y,
                                                self.xs: x
                                                })
        self.global_counter +=1
        if self.global_counter%10==0:
            self.cost_his.append(self.cost)

    def plot_cost(self):
        """
        This function will plot cost histgram
        :return:
        """
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('Training Steps')
        plt.grid()
        # plt.show()
        plt.savefig('cost.png')


    def restore_parameters(self):
        """
        This function will restore weights
        :return:
        """
        self.saver.restore(self.sess, self.dir_path + '/weights_saved/model.ckpt')  # restore model

    def save_parameters(self):
        """
        This function will save weights
        :return:
        """
        saver = tf.train.Saver()
        if not os.path.exists(self.dir_path+"/weights_saved"):
            os.mkdir(self.dir_path + "/weights_saved")
        saver_path = saver.save(self.sess, self.dir_path+'/weights_saved/model.ckpt')  # save model into save/model.ckpt file
        print('Model saved in file:', saver_path)



