import MyUtils
import input_queue
import os
import numpy as np

def TestRead(order,episode_len=6):

    if order == "omniglot":
        if not os.path.exists("train.npz") or not os.path.exists("test.npz"):
            MyUtils.read_omniglot()
        input_path = "train.npz"
        valid_path = "test.npz"

    else:
        raise NotImplementedError

    if order == "omniglot":
        input_size = (episode_len, 28, 28, 1)
    else:
        raise NotImplementedError

    with open(input_path, "rb") as f:
        input_npz = np.load(f)
        inputs = {}
        print('train',np.shape(input_npz.files))
        for filename in input_npz.files:
            inputs[filename] = input_npz[filename]

    with open(valid_path, "rb") as f:
        valid_npz = np.load(f)
        valid_inputs = {}
        print('test', np.shape(valid_npz.files))
        for filename in valid_npz.files:
            valid_inputs[filename] = valid_npz[filename]  # filename is the class label ,each class has 20 samples

def BaseTest():
    a=np.array([1,2,3,4])
    b=np.array([4,5,6,7])
    c=np.array([5,6,7,8])
    d=np.asarray([a,b,c])


    np.savez('testA',*(a,b,c))


def test():
    BaseTest()
    loader=np.load('testA.npz')
    for files in loader.files:
        print(files)
        print(loader[files])
if __name__ == '__main__':


    TestRead("omniglot")