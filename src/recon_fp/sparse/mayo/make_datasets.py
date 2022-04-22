'''
Make training dataset from mayo
'''

# %%
import os
import argparse
import sys
import numpy as np
import SimpleITK as sitk
import glob
import h5py

# import matplotlib.pyplot as plt

import ct_projector.projector.numpy as ct_base
import ct_projector.projector.numpy.parallel as ct_proj
import ct_projector.projector.numpy.fan_equiangluar as ct_fan

import recon_fp.utils.utils as utils

from recon_fp.sparse.locations import input_data_dir, working_dir


# %%
def get_args(default_args=[]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='lowdoseCTsets')
    parser.add_argument('--input_geometry', default='lowdoseCTsets/geometry.cfg')
    parser.add_argument('--output_dir', default='mayo/data/')
    parser.add_argument('--output_geometry', default='data/geometry_para.cfg')
    parser.add_argument('--nview', type=int, default=2304)

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
def save_results(img, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    sitk_img = sitk.GetImageFromArray(img)
    sitk.WriteImage(sitk_img, filename)


# %%
def main(args):
    ct_base.set_device(args.device)

    # prepare reconstruction
    projector_fan = ct_base.ct_projector()
    projector_fan.from_file(os.path.join(input_data_dir, args.input_geometry))
    projector_fan.nz = 1
    projector_fan.dz = 1
    projector_fan.nv = 1
    projector_fan.dv = 1
    angles_fan = projector_fan.get_angles()

    # prepare parallel beam
    projector_para = ct_base.ct_projector()
    projector_para.from_file(args.output_geometry)
    projector_para.nz = 1
    projector_fan.dz = 1
    projector_para.nv = 1
    projector_fan.dv = 1
    projector_para.nview = args.nview
    angles_para = projector_para.get_angles()

    filenames = glob.glob(os.path.join(input_data_dir, args.input_dir, '*_full_sino.mat'))
    print('Total files', len(filenames))
    for ifile, filename in enumerate(filenames):
        name = os.path.basename(filename).split('_')[0]
        print(ifile, name, flush=True)

        with h5py.File(filename, 'r') as f:
            prjs = np.copy(f['sino']).transpose([1, 0, 2])[np.newaxis]
        prjs = np.copy(prjs, 'C')

        print('Reconstruction', flush=True)
        projector_fan.nv = prjs.shape[2]
        projector_fan.nz = prjs.shape[2]
        fprj = ct_fan.ramp_filter(projector_fan, prjs, 'rl')
        refs = ct_fan.fbp_bp(projector_fan, fprj, angles_fan)
        refs = utils.mask_fov_fan(projector_fan, refs).astype(np.float32)

        print('Forward projection', flush=True)
        projector_para.nv = prjs.shape[2]
        projector_para.nz = prjs.shape[2]
        fp = ct_proj.distance_driven_fp(projector_para, refs, angles_para)

        print('Reconstruction from FP', flush=True)
        projector_para.nv = prjs.shape[2]
        projector_para.nz = prjs.shape[2]
        fprj = ct_proj.ramp_filter(projector_para, fp, 'hann')
        recons = ct_proj.distance_driven_bp(projector_para, fprj, angles_para, True)
        recons = utils.mask_fov_para(projector_para, recons).astype(np.float32)

        print('Saving', flush=True)
        output_dir = os.path.join(working_dir, args.output_dir, str(args.nview))
        os.makedirs(output_dir, exist_ok=True)

        # write configuration
        with open(os.path.join(output_dir, 'config.cfg'), 'w') as f:
            f.write(__file__ + '\n')
            for k in vars(args):
                f.write('{0} = {1}\n'.format(k, getattr(args, k)))

        save_results(
            fp[0],
            os.path.join(output_dir, name + '.prj.nii.gz')
        )
        save_results(
            (recons[0] / 0.019 * 1000).astype(np.int16),
            os.path.join(output_dir, name + '.nii.gz')
        )

        print('Done', flush=True)


# %%
if __name__ == '__main__':
    args = get_args([
        '--nview', '144'
    ])
    main(args)
