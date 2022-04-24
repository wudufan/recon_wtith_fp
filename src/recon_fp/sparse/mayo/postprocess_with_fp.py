'''
Postprocess the UNet results with the recovered forward projection
'''

# %%
import argparse
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import cupy as cp
import SimpleITK as sitk

from recon_fp.sparse.locations import working_dir

import ct_projector.projector.cupy as ct_base
import ct_projector.projector.cupy.parallel as ct_proj
import ct_projector.recon.cupy as ct_recon
import ct_projector.prior.cupy as ct_prior


# %%
def get_args(default_args=[]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='mayo/train/144/l2_depth_4/valid/')
    parser.add_argument('--prj_dir', default='mayo/data/144/')
    parser.add_argument('--output_dir', default='mayo/train/144/l2_depth_4/prj_recon/')
    parser.add_argument('--name', default='L291')
    parser.add_argument('--postfix', default='.nii.gz')
    parser.add_argument('--geometry', default='./data/geometry_para.cfg')
    parser.add_argument('--nview', type=int, default=144)

    parser.add_argument('--islice', type=int, nargs=2, default=None)

    parser.add_argument('--norm', type=float, default=1000)
    parser.add_argument('--offset', type=float, default=-1)

    parser.add_argument('--recover_niters', type=int, default=1500)
    parser.add_argument('--recover_stepsize', type=float, default=1)
    parser.add_argument('--recover_alpha', type=float, default=0.999)
    parser.add_argument('--recover_init', default='pred', choices=['zero', 'fp', 'pred', 'truth'])

    parser.add_argument('--post_niters', type=int, default=100)
    parser.add_argument('--post_nsubsets', type=int, default=12)
    parser.add_argument('--post_nesterov', type=float, default=0.5)

    parser.add_argument('--device', type=int, default=0)

    if 'ipykernel' in sys.argv[0]:
        args = parser.parse_args(default_args)
        args.debug = True
    else:
        args = parser.parse_args()
        args.debug = False

    for k in vars(args):
        print(k, '=', getattr(args, k), flush=True)

    return args


# %%
def load_data(args):
    x = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(
        working_dir, args.input_dir, args.name + '.x' + args.postfix
    )))
    y = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(
        working_dir, args.input_dir, args.name + '.y' + args.postfix
    )))
    pred = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(
        working_dir, args.input_dir, args.name + '.pred' + args.postfix
    )))
    prj = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(
        working_dir, args.prj_dir, args.name + '.prj' + args.postfix
    )))

    x = (x / args.norm - args.offset).astype(np.float32)[np.newaxis]
    y = (y / args.norm - args.offset).astype(np.float32)[np.newaxis]
    pred = (pred / args.norm - args.offset).astype(np.float32)[np.newaxis]
    prj = (prj / 0.019).astype(np.float32)[np.newaxis]

    # move to cupy
    x = cp.array(x, order='C')
    y = cp.array(y, order='C')
    pred = cp.array(pred, order='C')
    prj = cp.array(prj, order='C')

    return x, y, pred, prj


def load_projector(args):
    projector = ct_proj.ct_projector()
    projector.from_file(args.geometry)
    projector.nv = 1
    projector.nz = 1
    projector.nview = args.nview
    angles = projector.get_angles()

    angles = cp.array(angles, order='C')

    return projector, angles


def show_imgs(img_list, figsize, nrow, ncol, vmin=0.84, vmax=1.24, margin=96):
    '''
    img_list: if an element is np.array, show the image with given window;
    If an element is a tuple, the second element in the tuple is (vmin, vmax, margin)
    '''

    fig = plt.figure(figsize=figsize)
    for i in range(len(img_list)):
        plt.subplot(nrow, ncol, i + 1)
        if isinstance(img_list[i], list) or isinstance(img_list[i], tuple):
            img = img_list[i][0]
            vmin_show = img_list[i][1][0]
            vmax_show = img_list[i][1][1]
            m = img_list[i][1][2]
        else:
            img = img_list[i]
            vmin_show = vmin
            vmax_show = vmax
            m = margin
        plt.imshow(
            img[m:img.shape[0] - m, m:img.shape[1] - m].get(),
            'gray',
            vmin=vmin_show,
            vmax=vmax_show,
            aspect='auto'
        )
    plt.show()

    return fig


# %%
def mask_fov(projector: ct_base.ct_projector, img: cp.array):
    fov_size = projector.du * projector.nu / projector.dx

    xx, yy = np.meshgrid(np.arange(0, img.shape[2]), np.arange(0, img.shape[3]))
    cx = img.shape[2] / 2
    cy = img.shape[3] / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = np.where(dist < fov_size / 2, 1, 0)
    mask = cp.array(mask, cp.float32)
    img = img * mask[np.newaxis, np.newaxis]

    return img.astype(cp.float32)


# %%
def calc_fbp_matrix_norm(
    projector: ct_base.ct_projector,
    cuangles: cp.array,
    niter: int = 10,
    seed: int = 0
):
    cp.random.seed(seed)
    x = cp.random.uniform(size=[1, len(cuangles), projector.nv, projector.nu])
    x = x.astype(cp.float32)
    x = x / cp.linalg.norm(x)

    for i in range(niter):
        print(i, end=',', flush=True)
        fx = ct_proj.ramp_filter(projector, x, 'rl')
        recon = ct_proj.distance_driven_bp(projector, fx, cuangles, True)
        recon = mask_fov(projector, recon)
        norm = cp.linalg.norm(recon)

        fp = ct_proj.distance_driven_fp(projector, recon, cuangles)
        fp = weight_dd_fp_for_fbp(fp, cuangles, projector.dx, projector.dy)
        x = ct_proj.ramp_filter(projector, fp, 'rl')
        x = x / cp.linalg.norm(x)

        # print(
        #     cp.linalg.norm(fx),
        #     cp.linalg.norm(recon),
        #     cp.linalg.norm(fp),
        #     cp.linalg.norm(x),
        # )
    print('')

    return norm


def weight_dd_fp_for_fbp(cuprjs: cp.array, cuangles: cp.array, dx: float, dy: float):
    '''
    Apply weight to the distance driven fp to make it the transpose of FBP backprojector.
    '''

    cos_angles = cp.abs(cp.cos(cuangles))
    sin_angles = cp.abs(cp.sin(cuangles))
    wy = dy / cos_angles
    wx = dx / sin_angles

    weights = cp.ones(len(cuangles))
    weights[cos_angles > sin_angles] = wy[cos_angles > sin_angles]
    weights[cos_angles <= sin_angles] = wx[cos_angles <= sin_angles]

    weights = weights.astype(cp.float32)
    cuprjs = cuprjs / weights[cp.newaxis, :, cp.newaxis, cp.newaxis]

    return cuprjs.astype(cp.float32)


def recover_projection_from_img(
    projector: ct_base.ct_projector,
    cuprj0: cp.array,
    cuimg: cp.array,
    cuangles: cp.array,
    niters: int = 1000,
    stepsize: float = 1,
    alpha: float = 1,
    nesterov: float = 0.5
):
    mat_norm = calc_fbp_matrix_norm(projector, cuangles, niter=10)
    print(mat_norm)

    cuprj = cp.copy(cuprj0, 'C')
    cunesterov = cp.copy(cuprj0, 'C')

    for i in range(niters):
        cufprj = ct_proj.ramp_filter(projector, cunesterov, 'rl')
        curecon = ct_proj.distance_driven_bp(projector, cufprj, cuangles, True)
        curecon = mask_fov(projector, curecon)

        cudiff = curecon - cuimg

        cufp_diff = ct_proj.distance_driven_fp(projector, cudiff, cuangles)
        cufp_diff = weight_dd_fp_for_fbp(cufp_diff, cuangles, projector.dx, projector.dy)
        cuprj_diff = ct_proj.ramp_filter(projector, cufp_diff, 'rl')

        cuprj_new = cunesterov - cuprj_diff * stepsize / mat_norm / mat_norm
        stepsize *= alpha

        cunesterov = cuprj_new + nesterov * (cuprj_new - cuprj)
        cuprj = cuprj_new

        if (i + 1) % 100 == 0:
            print(i + 1, cp.sqrt(cp.mean(cudiff**2)))

    return cuprj


# %%
def post_ir(
    projector: ct_base.ct_projector,
    img0: cp.array,
    prjs: cp.array,
    angles: cp.array,
    nsubsets: int,
    niters: int = 1000,
    nesterov: float = 0.5
):
    projector.set_projector(ct_proj.distance_driven_fp, angles=angles)
    projector.set_backprojector(ct_proj.distance_driven_bp, angles=angles, is_fbp=False)

    projector_norm = projector.calc_projector_norm()
    norm_img = projector.calc_norm_img() / projector_norm / projector_norm

    recon = cp.copy(img0, 'C')
    recon_nesterov = cp.copy(recon, 'C')

    for i in range(niters):
        for isubset in range(nsubsets):
            # get subset
            inds = np.arange(isubset, len(angles), nsubsets)
            angles_current = cp.copy(angles[inds], 'C')
            prjs_current = cp.copy(prjs[:, inds, ...], 'C')

            recon_new = ct_recon.sqs_one_step(
                projector,
                recon_nesterov,
                prjs_current,
                norm_img,
                projector_norm,
                0,
                ct_prior.tv_sqs,
                {'weights': [1, 1, 1]},
                nsubsets,
                {'angles': angles_current},
                {'angles': angles_current},
                return_loss=False
            )

            recon_nesterov = recon_new + nesterov * (recon_new - recon)
            recon = recon_new

        # calculate loss
        _, data_loss, tv_loss = ct_recon.sqs_one_step(
            projector,
            recon,
            prjs,
            norm_img,
            projector_norm,
            0,
            ct_prior.tv_sqs,
            {'weights': [1, 1, 1]},
            return_loss=True
        )

        if (i + 1) % 100 == 0:
            print(i + 1, data_loss, tv_loss)

    return recon


# %%
def calc_rmse(x, y):
    return cp.sqrt(cp.mean((x - y)**2))


# %%
def hann_filter(cuprj):
    '''
    Apply the Hann filter window on the projection
    '''
    prj = cuprj.get()
    filter_len = 2 * prj.shape[-1] - 1
    hann_freq = 0.5 + 0.5 * np.cos(2 * np.pi * np.arange(filter_len) / filter_len)

    # padding
    pad_shape = list(prj.shape)
    pad_shape[-1] = prj.shape[-1] - 1
    prj_pad = np.concatenate([prj, np.zeros(pad_shape)], -1)

    # fft domain filter
    prj_pad_freq = np.fft.fft(prj_pad, axis=-1)
    prj_pad_freq *= hann_freq
    prj_pad = np.fft.ifft(prj_pad_freq, axis=-1)
    prj_hann = prj_pad[..., :prj.shape[-1]]
    prj_hann = np.real(prj_hann).astype(np.float32)

    return cp.array(prj_hann, order='C')


# %%
def process_slice_data(args, projector, angles, x, y, pred, prj):
    prj = hann_filter(prj)

    if args.debug:
        show_imgs(
            [x[0, x.shape[1] // 2],
             y[0, y.shape[1] // 2],
             pred[0, pred.shape[1] // 2],
             (prj[0, :, prj.shape[2] // 2, :], (None, None, 0))],
            (16, 16),
            2,
            2
        )

    # compare forward projection
    fp_x = ct_proj.distance_driven_fp(projector, x, angles)
    fp_pred = ct_proj.distance_driven_fp(projector, pred, angles)
    if args.debug:
        show_imgs(
            [((fp_x - prj)[0, :, fp_x.shape[2] // 2, :], (-1, 1, 0)),
             ((fp_pred - prj)[0, :, fp_x.shape[2] // 2, :], (-1, 1, 0))],
            (16, 8), 1, 2
        )

    # recover projection
    print('Recover projection...', flush=True)
    if args.recover_init == 'zero':
        prj_recon = cp.zeros(prj.shape, cp.float32)
    elif args.recover_init == 'fp':
        prj_recon = cp.copy(fp_x)
    elif args.recover_init == 'truth':
        prj_recon = cp.copy(prj)
    else:
        prj_recon = cp.copy(fp_pred)
    prj_recon = recover_projection_from_img(
        projector,
        prj_recon,
        x,
        angles,
        stepsize=args.recover_stepsize,
        niters=args.recover_niters,
        alpha=args.recover_alpha
    )
    if args.debug:
        print('FP prj RMSE', calc_rmse(prj, fp_x))
        print('Pred prj RMSE', calc_rmse(prj, fp_pred))
        print('Recover prj RMSE', calc_rmse(prj, prj_recon))
        show_imgs(
            [((prj_recon - prj)[0, :, prj_recon.shape[2] // 2, :], (-1, 1, 0))],
            (8, 8), 1, 1
        )

    # post-iteration with the recovered projection
    print('Post IR...', flush=True)
    recon = post_ir(projector, pred, prj_recon, angles, args.post_nsubsets, args.post_niters, args.post_nesterov)
    if args.debug:
        print('Pred RMSE', calc_rmse(pred, y), flush=True)
        print('Post IR RMSE', calc_rmse(recon, y), flush=True)
        show_imgs(
            [x[0, x.shape[1] // 2],
             y[0, y.shape[1] // 2],
             pred[0, pred.shape[1] // 2],
             recon[0, recon.shape[1] // 2]],
            (16, 16),
            2,
            2
        )

    return recon, prj_recon


# %%
def main(args):
    cp.random.seed(0)
    cp.cuda.Device(args.device).use()
    ct_base.set_device(args.device)

    print('Loading results...', flush=True)
    x_all, y_all, pred_all, prj_all = load_data(args)
    projector, angles = load_projector(args)

    # process slice by slice
    if args.islice is None:
        args.islice = [0, x_all.shape[1]]
    print('Slices {0} to {1}'.format(args.islice[0], args.islice[1]), flush=True)

    recon_all = []
    prj_recon_all = []
    for islice in range(args.islice[0], args.islice[1]):
        print('Processing slice {0}...'.format(islice), flush=True)

        # retrieve slice data
        x = cp.copy(x_all[:, [islice], :, :], 'C')
        y = cp.copy(y_all[:, [islice], :, :], 'C')
        pred = cp.copy(pred_all[:, [islice], :, :], 'C')
        prj = cp.copy(prj_all[:, :, [islice], :], 'C')

        recon, prj_recon = process_slice_data(args, projector, angles, x, y, pred, prj)

        recon_all.append(recon)
        prj_recon_all.append(prj_recon)
    recon_all = cp.concatenate(recon_all, 1)
    prj_recon_all = cp.concatenate(prj_recon_all, 2)

    # save results
    output_dir = os.path.join(working_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    # save config file
    with open(os.path.join(output_dir, 'running_params.log'), 'w') as f:
        f.write(__file__ + '\n')
        for k in vars(args):
            f.write('{0} = {1}\n'.format(k, getattr(args, k)))
    # save images
    recon_np = ((recon_all.get()[0] + args.offset) * args.norm).astype(np.int16)
    sitk_recon = sitk.GetImageFromArray(recon_np)
    sitk.WriteImage(sitk_recon, os.path.join(output_dir, args.name + '.postir.nii.gz'))
    # save recovered projection
    prj_recon_np = (prj_recon_all.get()[0] * 0.019).astype(np.float32)
    sitk_prj_recon = sitk.GetImageFromArray(prj_recon_np)
    sitk.WriteImage(sitk_prj_recon, os.path.join(output_dir, args.name + '.prj_recon.nii.gz'))


# %%
if __name__ == '__main__':
    nview = 288
    tag = 'l2_depth_4'

    args = get_args([
        '--recover_init', 'pred',
        '--recover_niters', '2000',
        '--recover_stepsize', '1',
        '--recover_alpha', '0.9999',
        '--post_niters', '100',
        '--post_nsubsets', '12',
        '--input_dir', f'mayo/train/{nview}/{tag}/valid/',
        '--prj_dir', f'mayo/data/{nview}/',
        '--output_dir', f'mayo/train/{nview}/{tag}/prj_recon/',
        '--nview', f'{nview}',
        # '--name', 'L291',
        # '--islice', '90', '96',
        # '--name', 'L143',
        # '--islice', '253',
        '--name', 'ACR',
        '--islice', '95', '96',
        # '--name', 'L067',
        # '--islice', '95',
    ])
    res = main(args)
