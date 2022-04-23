'''
Callback functions
'''

import tensorflow as tf
import numpy as np
import SimpleITK as sitk
import os

from typing import Callable


class SaveValid2DImageCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        model: tf.keras.Model,
        x: dict,
        y: dict,
        output_dir: str,
        interval: int = 1,
        postprocess: Callable[[np.array], np.array] = None,
        norm_x: float = 1000,
        norm_y: float = 1000,
        save_all_channels: bool = False
    ):
        '''
        x and y should be [nbatch, ny, nx, channel]
        '''
        super().__init__()

        self.model = model
        self.x = x
        self.y = y
        self.output_dir = output_dir
        self.interval = interval
        self.postprocess = postprocess
        self.norm_x = norm_x
        self.norm_y = norm_y
        self.save_all_channels = save_all_channels

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval != 0:
            return

        os.makedirs(self.output_dir, exist_ok=True)

        # predict all the x
        for name in self.x:
            preds = self.model.predict(self.x[name])
            if self.postprocess is not None:
                preds = self.postprocess(preds)

            xs = (self.x[name] * self.norm_x).astype(np.int16)
            preds = (preds * self.norm_y).astype(np.int16)

            sitk.WriteImage(
                sitk.GetImageFromArray(xs[..., 0]),
                os.path.join(self.output_dir, name + '.x.nii.gz')
            )
            sitk.WriteImage(
                sitk.GetImageFromArray(preds[..., 0]),
                os.path.join(self.output_dir, name + '.pred.nii.gz')
            )
            if self.save_all_channels:
                for i in range(1, xs.shape[-1]):
                    sitk.WriteImage(
                        sitk.GetImageFromArray(xs[..., i]),
                        os.path.join(self.output_dir, name + '.x-{0}.nii.gz'.format(i))
                    )
                for i in range(1, preds.shape[-1]):
                    sitk.WriteImage(
                        sitk.GetImageFromArray(preds[..., i]),
                        os.path.join(self.output_dir, name + '.pred-{0}.nii.gz'.format(i))
                    )

        # predict y
        if self.y is not None:
            for name in self.y:
                if self.postprocess is not None:
                    ys = self.postprocess(self.y[name])
                else:
                    ys = self.y[name]
                ys = (ys * self.norm_y).astype(np.int16)

                sitk.WriteImage(
                    sitk.GetImageFromArray(ys[..., 0]),
                    os.path.join(self.output_dir, name + '.y.nii.gz')
                )
                if self.save_all_channels:
                    for i in range(1, ys.shape[-1]):
                        sitk.WriteImage(
                            sitk.GetImageFromArray(ys[..., i]),
                            os.path.join(self.output_dir, name + '.y-{0}.nii.gz'.format(i))
                        )


class TensorboardSnapshotCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        model: tf.keras.Model,
        file_writer,
        x: dict,
        y: dict,
        ref: dict = None,
        interval: int = 1,
        postprocess: Callable[[np.array], np.array] = None,
        display_all_channels: bool = False,
        norm_x: float = 1000,
        vmin_x: float = -160,
        vmax_x: float = 240,
        norm_y: float = 1000,
        vmin_y: float = -160,
        vmax_y: float = 240
    ):
        super().__init__()

        self.model = model
        self.file_writer = file_writer
        self.x = x
        self.y = y
        self.ref = ref
        self.interval = interval
        self.norm_x = norm_x
        self.vmin_x = vmin_x
        self.vmax_x = vmax_x
        self.norm_y = norm_y
        self.vmin_y = vmin_y
        self.vmax_y = vmax_y

        # postprocessing on the prediction
        self.postprocess = postprocess

        # to display all channels or not
        self.display_all_channels = display_all_channels

    def make_snapshot(self, img, norm, vmin, vmax):
        img = (img * norm - vmin) / (vmax - vmin)

        return img[..., [0]]

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval != 0:
            return

        # predict all the x
        if self.ref is not None:
            preds = {k: self.model.predict(self.x[k]) - self.ref[k] for k in self.x}
        else:
            preds = {k: self.model.predict(self.x[k]) for k in self.x}

        if self.postprocess is not None:
            ys = {k: self.postprocess(self.y[k]) for k in self.y}
            preds = {k: self.postprocess(preds[k]) for k in preds}

        with self.file_writer.as_default():
            for name in self.x:
                tf.summary.image(
                    name + '/x',
                    self.make_snapshot(self.x[name], self.norm_x, self.vmin_x, self.vmax_x),
                    step=epoch
                )
                tf.summary.image(
                    name + '/pred',
                    self.make_snapshot(preds[name], self.norm_y, self.vmin_y, self.vmax_y),
                    step=epoch
                )

                if self.display_all_channels:
                    for i in range(1, self.x[name].shape[-1]):
                        tf.summary.image(
                            name + '/x-{0}'.format(i),
                            self.make_snapshot(self.x[name], self.norm_x, self.vmin_x, self.vmax_x),
                            step=epoch
                        )
                    for i in range(1, preds[name].shape[-1]):
                        tf.summary.image(
                            name + '/pred-{0}'.format(i),
                            self.make_snapshot(preds[name], self.norm_y, self.vmin_y, self.vmax_y),
                            step=epoch
                        )

            if self.y is not None:
                if self.postprocess is not None:
                    ys = {k: self.postprocess(self.y[k]) for k in self.y}
                else:
                    ys = self.y

                for name in ys:
                    tf.summary.image(
                        name + '/y',
                        self.make_snapshot(ys[name], self.norm_y, self.vmin_y, self.vmax_y),
                        step=epoch
                    )

                    if self.display_all_channels:
                        for i in range(1, ys[name].shape[-1]):
                            tf.summary.image(
                                name + '/pred-{0}'.format(i),
                                self.make_snapshot(ys[name], self.norm_y, self.vmin_y, self.vmax_y),
                                step=epoch
                            )
