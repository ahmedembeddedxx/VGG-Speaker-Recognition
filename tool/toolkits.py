import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d

def initialize_GPU(args):
    # Initialize GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    return

def get_chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

def debug_generator(generator):
    import cv2
    G = next(generator)
    for i, img in enumerate(G[0]):
        path = '../sample/{}.jpg'.format(i)
        img = np.asarray(img[:,:,::-1] + 128.0, dtype='uint8')
        cv2.imwrite(path, img)

def set_mp(processes=8):
    import multiprocessing as mp

    def init_worker():
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    global pool
    try:
        pool.terminate()
    except:
        pass

    if processes:
        pool = mp.Pool(processes=processes, initializer=init_worker)
    else:
        pool = None
    return pool

def get_vggface2_imglist(args):
    def get_datalist(s):
        with open(s, 'r') as file:
            datalist = file.readlines()
        imglist = []
        labellist = []
        for i in datalist:
            linesplit = i.split(' ')
            imglist.append(linesplit[0])
            labellist.append(int(linesplit[1].strip()))
        return imglist, labellist

    print('==> calculating image lists...')
    # Prepare training data.
    imgs_list_trn, lbs_list_trn = get_datalist(args.trn_meta)
    imgs_list_trn = [os.path.join(args.data_path, i) for i in imgs_list_trn]
    imgs_list_trn = np.array(imgs_list_trn)
    lbs_list_trn = np.array(lbs_list_trn)

    # Prepare validation data.
    imgs_list_val, lbs_list_val = get_datalist(args.val_meta)
    imgs_list_val = [os.path.join(args.data_path, i) for i in imgs_list_val]
    imgs_list_val = np.array(imgs_list_val)
    lbs_list_val = np.array(lbs_list_val)

    return imgs_list_trn, lbs_list_trn, imgs_list_val, lbs_list_val

def get_imagenet_imglist(args, trn_meta_path='', val_meta_path=''):
    with open(trn_meta_path) as f:
        strings = f.readlines()
        trn_list = np.array([os.path.join(args.data_path, '/'.join(string.split()[0].split(os.sep)[-4:]))
                             for string in strings])
        trn_lb = np.array([int(string.split()[1]) for string in strings])

    with open(val_meta_path) as f:
        strings = f.readlines()
        val_list = np.array([os.path.join(args.data_path, '/'.join(string.split()[0].split(os.sep)[-4:]))
                             for string in strings])
        val_lb = np.array([int(string.split()[1]) for string in strings])
        
    return trn_list, trn_lb, val_list, val_lb

def get_voxceleb2_datalist(args, path):
    with open(path) as f:
        strings = f.readlines()
        audiolist = np.array([os.path.join(args.data_path, string.split()[0]) for string in strings])
        labellist = np.array([int(string.split()[1]) for string in strings])
        
    return audiolist, labellist

def calculate_eer(y, y_score):
    # y denotes groundtruth scores,
    # y_score denotes the prediction scores.
    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr, kind='linear')(x), 0., 1.)
    thresh = interp1d(fpr, thresholds, kind='linear')(eer)
    return eer, thresh

def sync_model(src_model, tgt_model):
    print('==> synchronizing the model weights.')
    params = {}
    for l in src_model.layers:
        params[l.name] = l.get_weights()

    for l in tgt_model.layers:
        if l.name in params and len(l.get_weights()) > 0:
            l.set_weights(params[l.name])
    return tgt_model
