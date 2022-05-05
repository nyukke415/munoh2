# -*- coding: utf-8 -*-
# last updated:<2022/02/01/Tue 11:46:24 from:tuttle-desktop>

import cloudpickle

def flatten(l):
    flattened_l = list()
    for elm in l:
        # if type(elm) == str:
        #     flattened_l.append(elm)
        # else:
        #     flattened_l += flatten(elm)
        if type(elm) == list:
            flattened_l += flatten(elm)
        else:
            flattened_l.append(elm)
    return flattened_l



# file functions ################
def load_txt(path):
    with open(path, "r") as f:
        txt = f.read()
    print("loaded:\t"+path)
    return txt

def save_txt(path, txt):
    with open(path, "w") as f:
        f.write(txt)
    print("saved:\t"+path)

def load_pickle(path):
    with open(path, "rb") as f:
        w2i = cloudpickle.loads(f.read())
    print("loaded:\t"+path)
    return w2i

def save_pickle(path, data):
    with open(path, "wb") as f:
        cloudpickle.dump(data, f)
    print("saved:\t"+path)
