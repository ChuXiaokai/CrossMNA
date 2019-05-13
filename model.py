# encoding: utf8
import tensorflow as tf

class MultiNetworkEmb(object):
    def __init__(self, num_of_nodes, batch_size, K, node_embedding,
                     num_layer, layer_embedding):
        # 参数矩阵
        self.embedding = tf.Variable(tf.truncated_normal([num_of_nodes, node_embedding], stddev=0.3))
        self.embedding = tf.clip_by_norm(self.embedding, clip_norm=1, axes=1)
        self.L_embedding = tf.Variable(tf.truncated_normal([num_layer + 1, layer_embedding],stddev=0.3))
        self.L_embedding = tf.clip_by_norm(self.L_embedding, clip_norm=1, axes=1)
        self.W = tf.Variable(tf.truncated_normal([node_embedding, layer_embedding], stddev=0.3))
        self.W = tf.clip_by_norm(self.W, clip_norm=1, axes=1)

        # f1()
        self.u_i = tf.placeholder(name='u_i', dtype=tf.int32, shape=[batch_size * (K + 1)])
        self.u_j = tf.placeholder(name='u_j', dtype=tf.int32, shape=[batch_size * (K + 1)])
        self.label = tf.placeholder(name='label', dtype=tf.float32, shape=[batch_size * (K + 1)])
        self.this_layer = tf.placeholder(name='this_layer', dtype=tf.int32, shape=[batch_size * (K + 1)])

        def f1():
            self.u_i_embedding = tf.nn.embedding_lookup(self.embedding,self.u_i)
            self.u_j_embedding = tf.nn.embedding_lookup(self.embedding, self.u_j)


            # step 2. W u
            self.u_i_embedding = tf.matmul(self.u_i_embedding, self.W)
            self.u_j_embedding = tf.matmul(self.u_j_embedding, self.W)

            # step 3. layer emb r
            self.l_i_embedding = tf.nn.embedding_lookup(self.L_embedding, self.this_layer)

            # step 4. r_i = u_i*W + l
            self.r_i = self.u_i_embedding + self.l_i_embedding
            self.r_j = self.u_j_embedding + self.l_i_embedding
            # self.r_j = self.u_j_embedding

            # step 6. 前半部分loss
            self.inner_product = tf.reduce_sum(self.r_i * self.r_j, axis=1)
            # self.inner_product = tf.reduce_sum(self.u_i_embedding * self.u_j_embedding, axis=1)

            # loss function
            loss = -tf.reduce_sum(tf.log_sigmoid(self.label * self.inner_product))
            return loss


        self.loss = f1()

