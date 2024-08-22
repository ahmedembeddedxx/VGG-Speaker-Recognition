from __future__ import absolute_import, print_function
import os
import sys
import numpy as np
from pathlib import Path
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from tensorflow.keras import backend as K

np.seterr(all='raise')

sys.path.append('../tool')
import toolkits

# ===========================================
#        Parse the argument
# ===========================================
parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--gpu', default='', type=str)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--data_path', default='/scratch/local/ssd/weidi/voxceleb2/dev/wav', type=str)
parser.add_argument('--multiprocess', default=12, type=int)
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=10, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--epochs', default=56, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--warmup_ratio', default=0, type=float)
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'], type=str)
parser.add_argument('--ohem_level', default=0, type=int,
                    help='pick hard samples from (ohem_level * batch_size) proposals, must be > 1')
args = parser.parse_args()

def main():

    # gpu configuration
    toolkits.initialize_GPU(args)

    import model
    import generator

    # ==================================
    #       Get Train/Val.
    # ==================================
    trnlist, trnlb = toolkits.get_voxceleb2_datalist(args, path='../meta/voxlb2_train.txt')
    vallist, vallb = toolkits.get_voxceleb2_datalist(args, path='../meta/voxlb2_val.txt')

    # construct the data generator.
    params = {'dim': (257, 250, 1),
              'mp_pooler': toolkits.set_mp(processes=args.multiprocess),
              'nfft': 512,
              'spec_len': 250,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 5994,
              'sampling_rate': 16000,
              'batch_size': args.batch_size,
              'shuffle': True,
              'normalize': True,
              }

    # Datasets
    partition = {'train': trnlist.flatten(), 'val': vallist.flatten()}
    labels = {'train': trnlb.flatten(), 'val': vallb.flatten()}

    # Generators
    trn_gen = generator.DataGenerator(partition['train'], labels['train'], **params)
    network = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                           num_class=params['n_classes'],
                                           mode='train', args=args)
    # ==> load pre-trained model ???
    mgpu = len(K.tensorflow_backend._get_available_gpus())

    if args.resume:
        print("Attempting to load", args.resume)
        if os.path.isfile(args.resume):
            if mgpu == 1:
                # by_name=True, skip_mismatch=True
                network.load_weights(os.path.join(args.resume), by_name=True, skip_mismatch=True)
            else:
                network.layers[mgpu + 1].load_weights(os.path.join(args.resume))
            print(f'Successfully loaded model {args.resume}.')
        else:
            print(f"No checkpoint found at '{args.resume}'")

    print(network.summary())
    print(f'GPU {args.gpu} is training {len(partition["train"])} images, classes: 0-{np.max(labels["train"])} '
          f'loss: {args.loss}, aggregation: {args.aggregation_mode}, ohemlevel: {args.ohem_level}')

    model_path, log_path = set_path(args)
    normal_lr = LearningRateScheduler(step_decay)
    tbcallbacks = TensorBoard(log_dir=log_path, histogram_freq=0, write_graph=True, write_images=False,
                              update_freq=args.batch_size * 16)
    callbacks = [ModelCheckpoint(Path(model_path) / 'weights-{epoch:02d}-{acc:.3f}.h5',
                                 monitor='loss',
                                 mode='min',
                                 save_best_only=True),
                 normal_lr, tbcallbacks]

    if args.ohem_level > 1:     # online hard negative mining will be used
        candidate_steps = int(len(partition['train']) // args.batch_size)
        iters_per_epoch = int(len(partition['train']) // (args.ohem_level * args.batch_size))

        ohem_generator = generator.OHEM_generator(network,
                                                  trn_gen,
                                                  candidate_steps,
                                                  args.ohem_level,
                                                  args.batch_size,
                                                  params['dim'],
                                                  params['n_classes']
                                                  )

        A = ohem_generator.next()   # for some reason, I need to warm up the generator

        network.fit_generator(generator.OHEM_generator(network, trn_gen, iters_per_epoch,
                                                       args.ohem_level, args.batch_size,
                                                       params['dim'], params['n_classes']),
                              steps_per_epoch=iters_per_epoch,
                              epochs=args.epochs,
                              max_queue_size=10,
                              callbacks=callbacks,
                              use_multiprocessing=False,
                              workers=1,
                              verbose=1)

    else:
        network.fit_generator(trn_gen,
                              steps_per_epoch=int(len(partition['train']) // args.batch_size),
                              epochs=args.epochs,
                              max_queue_size=10,
                              callbacks=callbacks,
                              use_multiprocessing=False,
                              workers=1,
                              verbose=1)


def step_decay(epoch):
    '''
    The learning rate begins at 10^initial_power,
    and decreases by a factor of 10 every step epochs.
    '''
    half_epoch = args.epochs // 2
    stage1, stage2, stage3 = int(half_epoch * 0.5), int(half_epoch * 0.8), half_epoch
    stage4 = stage3 + stage1
    stage5 = stage4 + (stage2 - stage1)
    stage6 = args.epochs

    if args.warmup_ratio:
        milestone = [2, stage1, stage2, stage3, stage4, stage5, stage6]
        gamma = [args.warmup_ratio, 1.0, 0.1, 0.01, 1.0, 0.1, 0.01]
    else:
        milestone = [stage1, stage2, stage3, stage4, stage5, stage6]
        gamma = [1.0, 0.1, 0.01, 1.0, 0.1, 0.01]

    lr = 0.005
    init_lr = args.lr
    stage = len(milestone)
    for s in range(stage):
        if epoch < milestone[s]:
            lr = init_lr * gamma[s]
            break
    print(f'Learning rate for epoch {epoch + 1} is {lr}.')
    return float(lr)


def set_path(args):
    date = datetime.datetime.now().strftime("%Y-%m-%d")

    if args.aggregation_mode == 'avg':
        exp_path = os.path.join(args.aggregation_mode + f'_{args.loss}',
                                f'{date}_{args.net}_bs{args.batch_size}_{args.optimizer}_'
                                f'lr{args.lr}_bdim{args.bottleneck_dim}_ohemlevel{args.ohem_level}')
    elif args.aggregation_mode == 'vlad':
        exp_path = os.path.join(args.aggregation_mode + f'_{args.loss}',
                                f'{date}_{args.net}_bs{args.batch_size}_{args.optimizer}_'
                                f'lr{args.lr}_vlad{args.vlad_cluster}_'
                                f'bdim{args.bottleneck_dim}_ohemlevel{args.ohem_level}')
    elif args.aggregation_mode == 'gvlad':
        exp_path = os.path.join(args.aggregation_mode + f'_{args.loss}',
                                f'{date}_{args.net}_bs{args.batch_size}_{args.optimizer}_'
                                f'lr{args.lr}_vlad{args.vlad_cluster}_'
                                f'ghost{args.ghost_cluster}_bdim{args.bottleneck_dim}_ohemlevel{args.ohem_level}')

    exp_path = Path(exp_path)
    exp_path.mkdir(parents=True, exist_ok=True)
    model_path = str(exp_path)
    log_path = str(exp_path / 'logs')
    return model_path, log_path


if __name__ == '__main__':
    main()
