import os
import glob

import scipy.misc
import numpy as np

from tqdm import tqdm


def read_omniglot():
    """Read omniglot dataset, save them to two npz file"""
    omniglot_train = '/Users/zhangyuhan/Downloads/DataSet/omniglot/python/images_background'
    omniglot_eval = '/Users/zhangyuhan/Downloads/DataSet/omniglot/python/images_evaluation'

    DataTrain = []
    DataTest=[]
    classTrain=glob.glob(omniglot_train + '/*')
    classTest = glob.glob(omniglot_eval + '/*')
    for cls in tqdm(classTrain):
        alphabets = glob.glob(cls + '/*')
        for a in alphabets:
            characters = glob.glob(a + '/*')
            raws = []
            for ch in characters:  # 20 iters
                raw = scipy.misc.imread(ch)
                raw = scipy.misc.imresize(raw, (28, 28))
                for dg in [0, 90, 180, 270]:  # augmentation
                    raw_rot = scipy.misc.imrotate(raw, dg)
                    raw_rot = raw_rot[:, :, np.newaxis]  # (28, 28, 1)
                    raw_rot = raw_rot.astype(np.float32) / 255.
                    raws.append(raw_rot)
            DataTrain.append(np.asarray(raws))

    for cls in tqdm(classTest):
        alphabets = glob.glob(cls + '/*')
        for a in alphabets:
            characters = glob.glob(a + '/*')
            raws = []
            for ch in characters:  # 20 iters
                raw = scipy.misc.imread(ch)
                raw = scipy.misc.imresize(raw, (28, 28))
                for dg in [0, 90, 180, 270]:  # augmentation
                    raw_rot = scipy.misc.imrotate(raw, dg)
                    raw_rot = raw_rot[:, :, np.newaxis]  # (28, 28, 1)
                    raw_rot = raw_rot.astype(np.float32) / 255.
                    raws.append(raw_rot)
            DataTest.append(np.asarray(raws))
    np.savez('train.npz', *DataTrain)
    np.savez('test.npz', *DataTest)


