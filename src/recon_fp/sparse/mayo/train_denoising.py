'''
Basic denoising training
'''

# %%
import sys
import os
import tensorflow as tf
import tensorflow.keras.backend as K
import shutil
import argparse
import glob
import SimpleITK as sitk
import numpy as np

import recon_fp.models as models
import recon_fp.utils.config_manager as config_manager

from recon_fp.sparse.locations import working_dir


# %%
def get_args(default_args=[]):
    parser = argparse.ArgumentParser()
    parser.add_argument('config')

    if 'ipykernel' in sys.argv[0]:
        args, config, train_args = config_manager.parse_config_with_extra_arguments(parser, default_args)
        args.debug = True
    else:
        args, config, train_args = config_manager.parse_config_with_extra_arguments(parser)
        args.debug = False

    print(args.config)
    for sec in train_args:
        print('[{0}]'.format(sec))
        for k in train_args[sec]:
            print(k, '=', train_args[sec][k])
        print('', flush=True)

    # write the config file to the output directory
    output_dir = os.path.join(working_dir, train_args['IO']['output_dir'], train_args['IO']['tag'])
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'running_params.cfg'), 'w') as f:
        f.write(__file__ + '\n')
        for k in vars(args):
            f.write('{0} = {1}\n'.format(k, getattr(args, k)))
    with open(os.path.join(output_dir, 'config.cfg'), 'w') as f:
        config.write(f)

    return train_args


def get_model(train_args) -> tf.keras.Model:
    # network
    network = models.UNet2D(**train_args['Network'])
    _ = network.build()
    network.model.summary(line_length=120)

    # optimizer
    optimizer = tf.keras.optimizers.Adam()
    network.model.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())

    return network.model


def get_augmentation(train_args) -> tf.keras.preprocessing.image.ImageDataGenerator:
    data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=train_args['Data']['flip'],
        vertical_flip=train_args['Data']['flip'],
    )

    return data_gen


def get_tensorboard(train_args, output_dir):
    # tensorboard
    log_dir = os.path.join(output_dir, 'log')
    if train_args['Train']['relog'] and os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    return tensorboard_callback


def get_lr(train_args):
    # learning rates
    def scheduler(epoch, lr):
        epoch_list = train_args['Train']['epochs']
        lr_list = train_args['Train']['lr']
        for i in range(len(epoch_list)):
            if epoch < epoch_list[i]:
                return lr_list[i]
        return lr_list[-1]

    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

    return lr_callback


def get_checkpoint_callback(train_args, output_dir, len_data):
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(output_dir, '{epoch}.h5'),
        save_freq=len_data * train_args['Train']['save_freq'],
        verbose=1
    )
    tmp_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(output_dir, 'tmp.h5'),
        save_freq=len_data,
        verbose=0
    )

    return checkpoint_callback, tmp_checkpoint_callback


def load_dataset(train_args):
    input_dir_x = os.path.join(working_dir, train_args['IO']['x_dir'])
    input_dir_y = os.path.join(working_dir, train_args['IO']['y_dir'])
    train_names = train_args['IO']['train']
    valid_names = train_args['IO']['valid']
    postfix = train_args['IO']['postfix']

    if valid_names is None:
        valid_names = []
    if train_names is None or len(train_names) == 0:
        filenames = glob.glob(os.path.join(input_dir_x, '*' + postfix))
        names = [os.path.basename(f)[:-len(postfix)] for f in filenames]
        names = [f for f in names if not f.endswith('.prj')]
        train_names = [f for f in names if f not in valid_names]

    print('Train', train_names, flush=True)
    print('Valid', valid_names, flush=True)

    norm = train_args['Data']['norm']
    offset = train_args['Data']['offset']

    train_x = []
    for name in train_names:
        img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(input_dir_x, name + postfix)))
        img = img.astype(np.float32)
        img = img / norm + offset
        train_x.append(img)
    train_x = np.concatenate(train_x)[..., np.newaxis]

    train_y = []
    for name in train_names:
        img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(input_dir_y, name + postfix)))
        img = img.astype(np.float32)
        img = img / norm + offset
        train_y.append(img)
    train_y = np.concatenate(train_y)[..., np.newaxis]

    valid_x = {}
    for name in valid_names:
        img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(input_dir_x, name + postfix)))
        img = img.astype(np.float32)
        img = img / norm + offset
        valid_x[name] = img[..., np.newaxis]

    valid_y = {}
    for name in valid_names:
        img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(input_dir_y, name + postfix)))
        img = img.astype(np.float32)
        img = img / norm + offset
        valid_y[name] = img[..., np.newaxis]

    return train_x, train_y, valid_x, valid_y


def get_snapshot_callback(train_args, model, output_dir, valid_x, valid_y):
    log_dir = os.path.join(output_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)

    # extract the snapshot slices
    snapshots_x = {}
    snapshots_y = {}
    for name in valid_x:
        total_slices = len(valid_x[name])
        islice = min(train_args['Display']['islice'], total_slices - 1)
        snapshots_x[name] = np.copy(valid_x[name][[islice]], 'C')
        snapshots_y[name] = np.copy(valid_y[name][[islice]], 'C')

    snapshot_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'snapshots'))
    display_args = train_args['Display']

    snapshot_callback = models.TensorboardSnapshotCallback(
        model,
        snapshot_writer,
        snapshots_x,
        snapshots_y,
        postprocess=None,
        norm_x=display_args['norm_x'],
        vmin_x=display_args['vmin_x'],
        vmax_x=display_args['vmax_x'],
        norm_y=display_args['norm_y'],
        vmin_y=display_args['vmin_y'],
        vmax_y=display_args['vmax_y'],
    )

    return snapshot_callback


def get_validation_callback(train_args, model, output_dir, valid_x, valid_y):
    display_args = train_args['Display']

    validation_callback = models.SaveValid2DImageCallback(
        model,
        valid_x,
        valid_y,
        os.path.join(output_dir, 'valid'),
        train_args['Train']['save_freq'],
        postprocess=None,
        norm_x=display_args['norm_x'],
        norm_y=display_args['norm_y']
    )

    return validation_callback


# %%
def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args['Train']['device']
    K.clear_session()

    seed = args['Train']['seed']
    np.random.seed(seed)
    tf.random.set_seed(seed)

    output_dir = os.path.join(working_dir, args['IO']['output_dir'], args['IO']['tag'])

    print('Setting up model', flush=True)
    model = get_model(args)
    datagen = get_augmentation(args)
    tf_callback = get_tensorboard(args, output_dir)
    lr_callback = get_lr(args)

    print('Loading data', flush=True)
    train_x, train_y, valid_x, valid_y = load_dataset(args)
    datagen_x = datagen.flow(train_x, batch_size=args['Train']['batch_size'], seed=seed)
    datagen_y = datagen.flow(train_y, batch_size=args['Train']['batch_size'], seed=seed)

    snapshot_callback = get_snapshot_callback(args, model, output_dir, valid_x, valid_y)
    checkpoint_callback, tmp_checkpoint_callback = get_checkpoint_callback(args, output_dir, len(datagen_x))
    validation_callback = get_validation_callback(args, model, output_dir, valid_x, valid_y)

    print('Training', flush=True)
    model.fit(
        zip(datagen_x, datagen_y),
        epochs=args['Train']['epochs'][-1],
        steps_per_epoch=len(datagen_x),
        verbose=args['Data']['verbose'],
        callbacks=[
            lr_callback,
            tf_callback,
            snapshot_callback,
            validation_callback,
            checkpoint_callback,
            tmp_checkpoint_callback
        ]
    )


# %%
if __name__ == '__main__':
    args = get_args([
        './config/l2_depth_4_large.cfg',
        '--IO.tag', '"debug"',
        '--IO.train', '["L291"]',
        '--IO.valid', '["L291"]',
        '--Train.lr', '[0.0001]',
        '--Train.epochs', '[200]',
        '--Network.use_bn', '0',
        '--Train.save_freq', '10',
        '--Train.batch_size', '8',
        '--Data.verbose', '1',
    ])
    main(args)

    # data_gen = get_augmentation(args)
    # train_x, train_y, valid_x, valid_y = load_dataset(args)
