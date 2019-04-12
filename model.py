import tensorflow as tf
import numpy as np
from itertools import combinations
from tensorflow.python import debug as tfdbg
import time

class GNN_FSL:
    def __init__(self,hparams, input_tensor, label_tensor, is_train):
        self.num_classes = hparams.n
        self.batch_size = hparams.batch_size
        self.seq_len = hparams.seq_len
        self.input_dim = hparams.input_dim
        self.num_gcn_blocks=hparams.num_gcn_blocks
        self.lr = hparams.lr
        self.hop=hparams.hop
        self.label_cut=hparams.label_cut


        # self.input_placeholder = tf.nn.l2_normalize(tf.cast(input_tensor, tf.float32),axis=-1)
        self.input_placeholder = tf.cast(input_tensor, tf.float32)
        self.label_placeholder = label_tensor
        self.is_train = is_train
        if self.is_train:
            self.global_step = tf.get_variable("global_step", initializer=0, trainable=False)
        else:
            self.global_step = None
        feed_label, target_label = tf.split(self.label_placeholder, [self.seq_len - 1, 1],axis=1)
        self.target_label = tf.reshape(target_label,shape=[-1])
        # self.target_label = target_label
        # self.target_label=tf.one_hot(self.target_label,depth=self.num_classes,dtype=tf.float32)
        feed_label_one_hot_without_target = tf.one_hot(feed_label,depth=self.num_classes,dtype=tf.float32)
        self.feed_label_one_hot_with_target = tf.concat([feed_label_one_hot_without_target, tf.fill([self.batch_size, 1, self.num_classes],1.0/self.num_classes)], axis=1)
        self.concated_input = tf.concat([self.input_placeholder, self.feed_label_one_hot_with_target], axis=2)

        data_store=self.input_placeholder
        label_store=self.feed_label_one_hot_with_target

        '''for test only'''
        # name = 'GCN_Blocks'
        # with tf.variable_scope(name):
        #     data_store, _, self.diff,label_store, propagation_store ,self.Lap,self.simi,self.cmpr= self._gcn_block(input_data=data_store,input_label=label_store,add_dim=self.num_classes, drop=False)




        for i in range(self.num_gcn_blocks):
            #是否公用相似度函数和感受野比例
            # name='GCN_Blocks'
            name=f"GCN_Block_{i}"
            with tf.variable_scope(name):
                _,data_store, label_store,_ = self._gcn_block(input_data=data_store,
                                                                             input_label=label_store,
                                                                             add_dim=int(self.input_dim/2))

        with tf.variable_scope('last_Block'):

            data_store, _,label_store, propagation_store = self._gcn_block(input_data=data_store,
                                                                         input_label=label_store,
                                                                         add_dim=self.num_classes)
        self.label_store=label_store
        if self.label_cut=='yes':
            print('use cut')
            self.predict_label = label_store[:, -1, :]
        else:self.predict_label=data_store[:,-1,:]
        self.propagation = propagation_store
        ce_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target_label,logits=self.predict_label))
        self.loss = ce_loss
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)
        self.accuracy = self._calc_accuracy()


    def _calc_accuracy(self):
        with tf.name_scope("accuracy"):
            self.predictions = tf.argmax(self.predict_label, 1, name="predictions", output_type=tf.int32)
            labels = self.target_label
            correct_predictions = tf.equal(self.predictions, labels)
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            return accuracy

    def _gcn_block(self,input_data,input_label,add_dim,drop=False):
        # input_data: B*T*D D=64 T is the sequeence_len
        # input_label: B*T*N
        # input_dim: D D=64
        # option='nn'
        # self.Difference1=Difference= self._get_Difference(input_data)

        Difference= self._get_DifferenceV2(input_data=input_data)
        adjacency,cmpr= self._similarity_func(batch_size=self.batch_size, seq_len=self.seq_len, Difference=Difference,
                                              drop=drop, input_dim=input_data.shape[-1],option='nn')
        #修改

        # Laplacian=self._get_Laplacian(adjacency=adjacency)
        Laplacian = adjacency
        # propagation=self._get_propagation(Laplacian=Laplacian)
        # propagation =tf.nn.softmax(Laplacian)
        propagation =Laplacian
        # self.simi=simi
        # propagation = adjacency
        data_out=tf.matmul(propagation,input_data)
        target_label=tf.matmul(propagation,input_label)
        target_label=target_label[:,-1:,:]
        # target_label=tf.matmul(propagation[:,-1:,:],input_label)
        origin_label=input_label[:,:-1,:]
        label_out=tf.concat([origin_label,target_label],axis=1)
        data_out=tf.nn.leaky_relu(data_out)
        data_out=self._add_nn_block(x=data_out,out_channel=add_dim)
        # data_out=tf.layers.batch_normalization(inputs=data_out, axis=-1)
        # data_out = tf.nn.leaky_relu(data_out)
        data_store = tf.concat([input_data, data_out], axis=-1)
        return data_out, data_store, label_out, propagation  # leaky relu
        # return data_out,data_store,Difference,label_out,propagation,Laplacian,adjacency,cmpr#leaky relu


    # def _get_Difference(self,input_data):
    #     return tf.nn.l2_normalize(tf.py_func(get_difference, [input_data], tf.float32),axis=-1)

    def _get_DifferenceV2(self,input_data):
        shape = input_data.shape
        Difference = []
        for i in range(shape[1]):
            Difference_1=[]
            for j in range(shape[1]):
                Difference_0 = input=tf.abs(input_data[:, i, :] - input_data[:, j, :])
                print('Difference_0:',Difference_0.shape)
                Difference_1.append(Difference_0)
            Difference_1=tf.stack(values=Difference_1,axis=1)
            print('Difference_1:', Difference_1.shape)
            Difference.append(Difference_1)
        last_out=tf.stack(values=Difference,axis=1)
        print('diff_v2,output:',last_out.shape)
        return last_out


    def _similarity_func(self,batch_size,seq_len,Difference,input_dim,drop=False,option='nn'):
        #(0,0,0,...,0)->0???
        #input_data shape when nn :B by (T*T) by D=64
        #input data shape when cnn :B by  T by T by D=64
        # option='nn'
        last_output=Difference
        # last_output=tf.transpose(Difference,perm=[0,3,2,1])
        # dig_0=tf.matrix_band_part(last_output, -1, 0)
        # last_output=last_output-dig_0
        test_temp = tf.zeros([batch_size, 0,input_dim])
        for i in range(0,seq_len-1):
            for j in range(i+1,seq_len):
                test_temp=tf.concat([test_temp,last_output[:,i:i+1,j,:]],axis=1)
        last_output=test_temp
        output_tocompare =test_temp
        print('last_output_0:',last_output.shape)
        # assert last_output.shape[1]==(self.seq_len*(self.seq_len-1)/2)


        if option=='cnn':
            last_output = tf.reshape(last_output, [batch_size, seq_len, seq_len, input_dim])
            for output_dim in [64,8,1]:

                with tf.variable_scope(f"similarN_{output_dim}",reuse=tf.AUTO_REUSE):

                    last_output =self._add_cnn_block(last_output,out_channel=output_dim)
                    if drop==True and output_dim>(int(input_dim)/4) and output_dim<(int(input_dim)/2) :last_output=tf.layers.dropout(inputs=last_output,rate=0.3)


        elif option=='nn':
            temp_dim=input_dim
            last_output= tf.reshape(last_output, [self.batch_size, int(seq_len*(seq_len-1)/2), temp_dim])
            # sup_similarity = []
            # for i in tf.unstack(last_output,axis=1):
            #     similarity_temp=i
            #     for output_dim in [128,16,1]:
            #         with tf.variable_scope(f"similarN_{output_dim}",reuse=tf.AUTO_REUSE):
            #             similarity_temp = self._add_nn_block(x=similarity_temp,out_channel=output_dim)
            #     sup_similarity.append(similarity_temp)
            # last_output=tf.squeeze(tf.stack(sup_similarity, axis=1))
            similarity_temp = last_output
            input_dim=last_output.shape[-1]
            for output_dim in [2*int(input_dim), int(input_dim)/2, 1]:
                with tf.variable_scope(f"similarN_{output_dim}", reuse=tf.AUTO_REUSE):
                    similarity_temp = self._add_nn_block(x=similarity_temp, out_channel=output_dim)
            print('after_nn:',last_output.shape)
            last_output=tf.squeeze(similarity_temp)

        else:
            last_output = tf.reshape(last_output, [self.batch_size, -1, input_dim])
            last_output=tf.reduce_sum(last_output,axis=-1)
        # print('last_output_1:', last_output.shape)
        last_output=tf.squeeze(last_output)
        # print('last_output_2:', last_output.shape)
        test_temp=tf.zeros([batch_size,0])
        temp_tocompare=tf.zeros([batch_size,0,input_dim])
        index=0
        for i in range(seq_len):
            test_temp=tf.concat([test_temp,tf.zeros([batch_size,i+1],tf.float32)],axis=-1)
            test_temp=tf.concat([test_temp,last_output[:,index:index+(seq_len-i-1)]],axis=-1)
            temp_tocompare=tf.concat([temp_tocompare,tf.zeros([batch_size,i+1,input_dim],tf.float32)],axis=-2)
            temp_tocompare=tf.concat([temp_tocompare,output_tocompare[:,index:index+(seq_len-i-1),:]],axis=-2)
            index=index+(seq_len-i-1)
        last_output=test_temp




        # print('last_output_3:',last_output.shape)
        last_output = tf.reshape(tensor=last_output, shape=[self.batch_size, self.seq_len, self.seq_len])
        temp_tocompare=tf.reshape(tensor=temp_tocompare, shape=[self.batch_size, self.seq_len, self.seq_len,input_dim])
        # print('last_output_4:', last_output.shape)
        last_output=last_output+tf.transpose(last_output,perm=[0,2,1])
        temp_tocompare=temp_tocompare+tf.transpose(temp_tocompare,perm=[0,2,1,3])
        sup_similarity = []
        index = 0
        for i in tf.unstack(last_output, axis=1):
            part0, zero0, part1 = tf.split(value=i, num_or_size_splits=[index, 1, seq_len - index - 1], axis=-1)
            part0, part1 = tf.split(value=tf.nn.softmax(tf.concat([part0, part1], axis=-1),axis=-1),
                                    num_or_size_splits=[index, seq_len - index - 1], axis=-1)
            similarity_temp = tf.concat([part0, zero0, part1], axis=-1)
            sup_similarity.append(similarity_temp)
            index=index+1
        last_output = tf.stack(sup_similarity, axis=1)
        # last_output=tf.nn.softmax(last_output,axis=-1)
        # last_output=last_output-tf.matrix_diag(tf.matrix_diag_part(last_output))
        return last_output,temp_tocompare


    def _get_Laplacian(self,adjacency):
        #因为假设全连接，将度矩阵设置为单位阵????

        # degreeMat=tf.eye(num_rows=self.seq_len,num_columns=self.seq_len,batch_shape=(self.batch_size,))
        # 修改度矩阵
        degreeget=tf.reduce_sum(adjacency,axis=-1)
        degreeMat = tf.matrix_diag(degreeget)
        assert degreeMat.get_shape()==adjacency.get_shape()
        Laplacian=degreeMat-adjacency

        return Laplacian


    def _get_propagation(self,Laplacian):
        # propagetion = tf.zeros([self.batch_size,self.seq_len,self.seq_len])
        theta_origin = tf.get_variable(
            initializer=tf.random_uniform(shape=(1,), minval=0, maxval=1, dtype=tf.float32, seed=None), name='theta_origin')
        propagetion =theta_origin*tf.eye(num_rows=self.seq_len,num_columns=self.seq_len,batch_shape=(self.batch_size,))
        assert propagetion.get_shape()==Laplacian.get_shape()
        tep_L=Laplacian
        for i in range(self.hop):
            exact_hop=i+1
            name=f"hop_{exact_hop}"
            with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
                hop_theta=tf.get_variable(initializer=tf.random_uniform(shape=(1,),minval=0, maxval=1, dtype=tf.float32, seed=None),name='theta')
                propagetion=propagetion+hop_theta*tep_L
                tep_L=tf.matmul(tep_L,Laplacian)
        propagetion = tf.nn.softmax(propagetion, axis=-1)
        return propagetion




    def _add_nn_block(self,x,out_channel):
        # weights = tf.tile(tf.expand_dims(tf.get_variable('weights1', shape=[in_channel, out_channel], dtype=tf.float32,
        #                           initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32)),0),[self.batch_size,1,1])
        # biases = tf.get_variable('biases1', shape=[out_channel], dtype=tf.float32,
        #                          initializer=tf.constant_initializer(0.1))
        # mul=tf.matmul(x,weights)+biases
        # self.l2_loss += tf.nn.l2_loss(weights)
        # self.l2_loss += tf.nn.l2_loss(biases)
        #
        # if activation=='relu':
        #     out_put=tf.nn.relu(mul)
        # elif activation=='sigmoid':
        #     out_put=tf.nn.sigmoid(mul)
        # else:
        #     out_put = tf.nn.tanh(mul)

        out_put=tf.layers.dense(inputs=x,units=out_channel,name='nn_block',use_bias=True,bias_initializer=tf.truncated_normal_initializer(stddev=0.01),activation=tf.nn.leaky_relu,reuse=tf.AUTO_REUSE)
        # mul = tf.layers.batch_normalization(inputs=mul, axis=-1)


        return out_put



    def _add_cnn_block(self,x,out_channel):


        cnn_out=tf.layers.conv2d(inputs=x,filters=out_channel,kernel_size=(3,3),strides=(1,1),activation=tf.nn.relu,name='cnn_block_conv',
                                 padding='same',data_format='channels_last',kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        block_out=tf.layers.batch_normalization(inputs=cnn_out,axis=-1,name='cnn_block_bn')
        return block_out




def get_difference(input_data):
    shape=np.shape(input_data)
    Difference=np.zeros([shape[0],shape[1],shape[1],shape[2]],dtype=np.float32)
    for i in range(shape[1]):
        for j in range(shape[1]):
            Difference[:, i, j, :] = np.abs(input_data[:, i, :] - input_data[:, j, :])
    return Difference





def _make_dummy_data(batch_size,seq_len,dim,num_class,k):

    input_data = np.random.randn(batch_size, seq_len, dim)
    # label_data = np.random.randint(num_class, size=(batch_size, seq_len))
    label_data = np.array([np.random.permutation(np.arange(num_class)) for i in range(batch_size)])
    label_taget=np.random.randint(num_class,size=(batch_size,1))
    label_data=np.concatenate((label_data,label_taget),axis=1)
    return input_data, label_data

def _GNN_test():
    class Dummy: pass

    hparams = Dummy()
    hparams.n = 3
    hparams.k=1
    hparams.input_dim = 10
    hparams.num_gcn_blocks=5
    hparams.batch_size = 20
    hparams.seq_len = hparams.n*hparams.k+1
    hparams.lr = 1e-3
    hparams.hop=1
    hparams.label_cut = 'yes'
    # np.set_printoptions(threshold='nan')  # 全部输出

    with tf.Graph().as_default():
        dummy_input, dummy_label = _make_dummy_data(batch_size=hparams.batch_size,seq_len=hparams.seq_len,dim=hparams.input_dim,num_class=hparams.n,k=hparams.k)
        print(np.shape(dummy_input),np.shape(dummy_label))
        print('dunmmy_input:\n',dummy_input)
        print('dummy label\n',dummy_label)
        model = GNN_FSL(hparams, tf.stack(dummy_input), tf.cast(tf.stack(dummy_label), tf.int32), True)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)


        with sess.as_default():
            # sess=tfdbg.LocalCLIDebugWrapperSession(sess)

            sess.run(tf.global_variables_initializer())
            # sess.run(init)
            '''查看参数'''
            a=[v for v in tf.trainable_variables()]
            for i in range(len(a)):
                print(a[i])
            # target_label=sess.run([model.target_label])
            # print(target_label)
            time_start = time.time()
            for i in range(1000):


                '''test only'''
                # Df1,Df2,_, train_data,loss, acc,target_label,predictlabel,propagation,Lap,simi,cmpr,feed_label,label_store= sess.run([model.Difference1,model.Difference2,model.train_step, model.concated_input,model.loss, model.accuracy,model.target_label,model.predict_label,\
                #                                                                                model.propagation,model.Lap,model.simi,model.cmpr,model.feed_label_one_hot_with_target,model.label_store])

                _, train_data, loss, acc, target_label, predictlabel= sess.run([model.train_step, model.concated_input, model.loss, model.accuracy, model.target_label,
                     model.predict_label])


                if i%100==0:
                    print('dunmmy_input:\n', dummy_input)
                    # print('Dif1\n',Df1)
                    # print('Dif2\n',Df2)
                    print('dummy label\n', dummy_label)
                    print("step is ", i, "\n", loss, acc, '\n')
                    print("predict is \n",predictlabel)
                    print('target is \n',target_label)
                    # print('feed_label\n',feed_label)
                    # print('label_store',label_store)

                    # # print('compare is \n', cmpr)
                    # print('simi is', '\n', simi, '\n')
                    # print('Lap is \n',Lap)
                    # print('propagation is','\n',propagation,'\n')



                # if acc>=0.99:break

            time_end = time.time()
            print('time cost', time_end - time_start, 's')

if __name__ == "__main__":
    _GNN_test()


