# encoding: utf8
from utils import *
from model import *
from AUC import *
from align_metric import *
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = ArgumentParser("network alignment",
                          formatter_class=ArgumentDefaultsHelpFormatter,
                          conflict_handler='resolve')
parser.add_argument('--task', default='NetworkAlignment', type=str)
parser.add_argument("--p", default=0.3, type=str)
parser.add_argument('--dataset', default='Twitter', type=str)
parser.add_argument('--node-dim', default=200, type=int)
parser.add_argument('--layer-dim', default=100, type=int)
parser.add_argument('--batch-size', default=512, type=int)
parser.add_argument('--neg-samples', default=5, type=int)
parser.add_argument('--output', default='node2vec.pk', type=str)
args = parser.parse_args()

dataset = args.dataset
p = args.p
print(dataset, p, args.task)
if args.task == 'NetworkAlignment':
    path = 'node_matching/'+dataset+'/'+'new_network'+str(p)+'.txt'
elif args.task == 'LinkPrediction':
    path = 'link_prediction/'+dataset+'/train'+str(p)+'.txt'


"""step 1. load data"""
layers, num_nodes, id2node = readfile(graph_path=path)
num_layers = len(layers.keys())

"""step 2. initial negative sampling table"""
for layerid in layers:
    g = layers[layerid]
    g.init_neg()

"""step 3. create model"""
model = MultiNetworkEmb(num_of_nodes=num_nodes, 
                        batch_size=args.batch_size, 
                        K=args.neg_samples,
                        node_embedding=args.node_dim,
                        num_layer=num_layers, 
                        layer_embedding=args.layer_dim
        ) 


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

"""step 4. start training session"""
with tf.Session(config=config) as sess:

    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.025)
    train_op = optimizer.minimize(model.loss)
    tf.global_variables_initializer().run()
    initial_embedding = sess.run(model.embedding)

    for epoch in range(100):
        t = time.clock()
        batches = gen_batches(layers, batch_size=args.batch_size, K=1)
        print("epoch {0}: time for generate batches={1}s".format(epoch, time.clock()-t))

        total_loss = 0.0
        t = time.clock()
        for batch in batches:
            u_i, u_j, label, this_layer = batch
            feed_dict = {model.u_i: u_i, model.u_j: u_j, model.label: label,
                         model.this_layer: this_layer,
                         }
            _, loss = sess.run([train_op, model.loss], feed_dict=feed_dict)
            total_loss += loss
        print("epoch {0}: time for training={1}, total_loss={2}s".format(epoch, time.clock()-t, total_loss))

        if epoch  ==  99:
            if args.task == 'NetworkAlignment':
                each_layer_nodes = {}
                inter_vectors = sess.run(model.embedding)

                node2vec = get_alignment_emb(inter_vectors, layers, id2node)

                # test multi-network alignment
                result = eval_emb('node_matching/' + args.dataset + '/networks' + str(args.p) + '.pk', node2vec, )

                # dump node2vec
                start_time = time.clock()
                pickle.dump(inter_vectors, open('emb/'+args.output, 'wb'))
                end_time = time.clock()
                print('epoch {0}: time for alignment {1}'.format(epoch, end_time-start_time))

            elif args.task == 'LinkPrediction':
                inter_vectors = sess.run(model.embedding)
                W = sess.run(model.W)
                layers_embedding = sess.run(model.L_embedding)

                node2vec = get_intra_emb(inter_vectors, W, layers_embedding, layers, id2node)

                # test link prediction
                auc = []
                for i in range(1, num_layers+1):
                    each_auc = []
                    for _ in range(5):
                        tmp_auc = AUC(node2vec[i], 'link_prediction/' + dataset + '/test' + str(p) + '.txt', i)
                        if tmp_auc:
                            each_auc.append(tmp_auc)
                    auc.append(np.mean(each_auc))
                print('epoch {0}: auc={1}'.format(epoch, np.mean(auc)))

                # dump node2vec
                pickle.dump([inter_vectors, W, layers_embedding], open('emb/'+args.output, 'wb'))

