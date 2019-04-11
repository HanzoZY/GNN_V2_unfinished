#CUDA_VISIBLE_DEVICES=2 python train.py 指定GPU


import argparse
import tensorflow as tf
import numpy as np
import os
import time
from model import GNN_FSL
from omniglot_embed import OmniglotEmbedNetwork
from input_queue import FewShotInputQueue
import MyUtils
from tensorflow.python import debug as tfdbg


def define_flags():
    flags = argparse.ArgumentParser()

    flags.add_argument("--n", type=int, default=None, help="N [Required]")
    flags.add_argument("--k", type=int, default=None, help="K [Required]")
    flags.add_argument("--dataset", type=str, default="omniglot", help="Dataset (omniglot / miniimage) [omniglot]")

    flags.add_argument("--num_gcn_blocks", type=int, default=5, help="List of dilation size[Required]")

    flags.add_argument("--batch_size", type=int, default=128, help="Batch size B[128]")
    flags.add_argument("--input_dim", type=int, default=64, help="Dimension of input D[64]")
    flags.add_argument("--lr", type=float, default=1e-3, help="Learning rate[1e-3]")
    flags.add_argument("--hop", type=int, default=1, help="hop default 1")
    flags.add_argument("--nn_option", type=str, default='nn', help="nn option")
    flags.add_argument("--label_cut", type=str, default='no', help="use label_cut")

    return flags.parse_args()


def train():
    hparams = define_flags()
    hparams.seq_len = episode_len = hparams.n * hparams.k + 1

    if hparams.dataset == "omniglot":
        if not os.path.exists("/youedata/home/zhangyuhan/Data/OmniglotNPZ/train.npz") or not os.path.exists(
                "/youedata/home/zhangyuhan/Data/OmniglotNPZ/test.npz"):
            MyUtils.read_omniglot()
        input_path = "/youedata/home/zhangyuhan/Data/OmniglotNPZ/train.npz"
        valid_path = "/youedata/home/zhangyuhan/Data/OmniglotNPZ/test.npz"

    else:
        raise NotImplementedError

    if hparams.dataset == "omniglot":
        input_size = (episode_len, 28, 28, 1)
    else:
        raise NotImplementedError

    with open(input_path, "rb") as f:
        input_npz = np.load(f)
        inputs = {}
        for filename in input_npz.files:
            inputs[filename] = input_npz[filename]

    with open(valid_path, "rb") as f:
        valid_npz = np.load(f)
        valid_inputs = {}
        for filename in valid_npz.files:
            valid_inputs[filename] = valid_npz[filename]   #filename is the class label ,each class has 20*4 samples

    with tf.Graph().as_default():
        q = FewShotInputQueue(inputs.keys(), inputs, hparams.n, hparams.k)
        valid_q = FewShotInputQueue(valid_inputs.keys(), valid_inputs, hparams.n, hparams.k)

        generated_input, generated_label = tf.py_func(q.make_one_data, [], [tf.float32, tf.int32])
        batch_tensors = tf.train.batch([generated_input, generated_label], batch_size=hparams.batch_size, num_threads=4,
                                       shapes=[input_size, (episode_len,)], capacity=hparams.batch_size*5)
        valid_input, valid_label = tf.py_func(valid_q.make_one_data, [], [tf.float32, tf.int32])
        valid_batch_tensors = tf.train.batch([valid_input, valid_label], batch_size=hparams.batch_size, num_threads=4,
                                             shapes=[input_size, (episode_len,)], capacity=hparams.batch_size*5)
        #each batch with the shape(128,)
        with tf.variable_scope("networks"):
            embed_network = OmniglotEmbedNetwork(batch_tensors, hparams.batch_size)
            gnn = GNN_FSL(hparams, embed_network.output, embed_network.label_placeholder, True)

        with tf.variable_scope("networks", reuse=True):
            valid_embed_network = OmniglotEmbedNetwork(valid_batch_tensors, hparams.batch_size)
            valid_gnn= GNN_FSL(hparams, valid_embed_network.output, valid_embed_network.label_placeholder, False)
        params_to_str = f"M_Nets_{hparams.input_dim}_{hparams.lr}_{hparams.n}_{hparams.k}_{hparams.num_gcn_blocks}_{hparams.label_cut}_{hparams.nn_option}"
        log_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", params_to_str))
        # Summaries
        tf.summary.scalar("train_loss", gnn.loss)
        tf.summary.scalar("train_acc", gnn.accuracy)

        tf.summary.scalar("valid_loss", valid_gnn.loss)
        tf.summary.scalar("valid_acc", valid_gnn.accuracy)

        tf.summary.image("inputs", valid_embed_network.input_placeholder[0], max_outputs=episode_len)


        # Supervisor
        supervisor = tf.train.Supervisor(
            logdir=log_dir,
            save_summaries_secs=240,
            save_model_secs=600,
            global_step=gnn.global_step,
        )


        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        print("Training start")
        #使用supervisor 有checkpoint就从checkpoint中载入，没有就自动初始化哦
        with supervisor.managed_session(config=config) as sess:
            min_dev_loss = 10000
            min_step = -1

            STEP_NUM = 10000000
            EARLY_STOP = 3000000
            print_every = 500

            HUGE_VALIDATION_CYCLE = print_every * 20

            # writer = tf.summary.FileWriter(logdir=log_dir,graph=sess.graph)
            # writer.close()
            last_dev = time.time()

            for step in range(STEP_NUM):
                if supervisor.should_stop():
                    break

                if step - min_step > EARLY_STOP:
                    print("Early stopping...")
                    break

                if step % print_every != 0:
                    _, loss, acc, global_step = sess.run(
                        [gnn.train_step, gnn.loss, gnn.accuracy, gnn.global_step])
                    # print(gnn.predict_label.shape)
                else:
                    _, loss, acc, global_step = sess.run(
                        [gnn.train_step, gnn.loss, gnn.accuracy, gnn.global_step])

                    loss, acc = sess.run([valid_gnn.loss, valid_gnn.accuracy])

                    current_time = time.time()
                    print(f'Evaluate(Step {step}/{global_step} : valid loss({loss}), acc({acc}) in {current_time - last_dev} s')

                    _, train_data, loss, acc, target_label, predictlabel, propagation, diff, Lap, simi, cmpr = sess.run(
                        [valid_gnn.train_step, valid_gnn.concated_input, valid_gnn.loss, valid_gnn.accuracy, valid_gnn.target_label,
                         valid_gnn.predict_label, \
                         valid_gnn.propagation, valid_gnn.diff, valid_gnn.Lap, valid_gnn.simi, valid_gnn.cmpr])

                    print("predict is \n", predictlabel)
                    print('target is \n', target_label)
                    # print('diff is: \n', diff, '\n')
                    # print('compare is \n', cmpr)
                    # print('simi is', '\n', simi, '\n')
                    print('Lap is \n', Lap)
                    print('propagation is', '\n', propagation, '\n')
                    print("loss and acc \n", loss, acc, '\n')












                    # HUGE VALIDATION
                    if step != 0 and step % HUGE_VALIDATION_CYCLE == 0:
                        total_loss = total_acc = 0.
                        BATCH_NUM = 40
                        for _ in range(BATCH_NUM):
                            loss, acc = sess.run([valid_gnn.loss, valid_gnn.accuracy])
                            total_loss += loss * hparams.batch_size
                            total_acc += acc * hparams.batch_size

                        total_loss /= BATCH_NUM * hparams.batch_size
                        total_acc /= BATCH_NUM * hparams.batch_size

                        huge_data_acc_summary = tf.Summary()
                        huge_data_acc_summary.value.add(tag="huge_data_accuracy", simple_value=total_acc)
                        supervisor.summary_computed(sess, huge_data_acc_summary, global_step=global_step)

                        huge_data_loss_summary = tf.Summary()
                        huge_data_loss_summary.value.add(tag="huge_data_loss", simple_value=total_loss)
                        supervisor.summary_computed(sess, huge_data_loss_summary, global_step=global_step)

                    last_dev = current_time

                    if loss < min_dev_loss:
                        min_dev_loss = loss
                        min_step = step


class h_test():
    def __init__(self):
        self.n=5
        self.k=1
        self.dataset="omniglot"
        self.num_gcn_blocks=3
        self.batch_size=64
        self.input_dim=64
        self.lr=1e-3
        self.reg_coeff=1e-3
        self.hop=1
        self.seq_len=0


if __name__ == "__main__":
    train()
