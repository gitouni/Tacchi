import os
import shutil
import torch
import torch.optim
import numpy as np
import torch.nn as nn
from tqdm import tqdm 
from ndp.nets import Deformation_Pyramid
from ndp.loss import compute_truncated_chamfer_distance
import yaml
# from main_utils import *
import argparse
import open3d as o3d
import copy
import shutil
def resfresh_dir(dirname:str):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname)


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=str,default="cfgs/NDP_pcd_flow.yml")
    parser.add_argument("--height_dir",type=str,default="res/height")
    parser.add_argument("--output_dir",type=str,default="res/flow")
    return parser.parse_args()

def draw_registration_result(source, target, transformation):
    if isinstance(source, np.ndarray):
        source_temp = o3d.geometry.PointCloud()
        source_temp.points = o3d.utility.Vector3dVector(source)
        target_temp = o3d.geometry.PointCloud()
        target_temp.points = o3d.utility.Vector3dVector(target)
    else:
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def NDP_register(model_args:dict, register_args:dict, src_tensor:torch.Tensor, tgt_tensor:torch.Tensor, debug=False):
    assert src_tensor.ndim == 2 and tgt_tensor.ndim == 2, "shape of input tensor must be (N, 3)"
    src_tensor_mean = src_tensor.mean(dim=0, keepdims=True)  # (N, 1)
    tgt_tensor_mean = tgt_tensor.mean(dim=0, keepdims=True)  # (N, 1)
    src_tensor_centered = src_tensor - src_tensor_mean
    tgt_tensor_centered = tgt_tensor - tgt_tensor_mean
    s_sample = src_tensor_centered.clone()
    t_sample = tgt_tensor_centered.clone()
    BCE = nn.BCELoss()
    model = Deformation_Pyramid(**model_args)
    for level in range(model.n_hierarchy):

        """freeze non-optimized level"""
        model.gradient_setup(optimized_level=level)

        optimizer = torch.optim.Adam(model.pyramid[level].parameters(), lr=register_args['lr'])

        break_counter = 0
        loss_prev = 1e+6

        """optimize current level"""
        for _ in range(register_args['iters']):


            s_sample_warped, data = model.warp(s_sample, max_level=level, min_level=level)

            loss = compute_truncated_chamfer_distance(s_sample_warped.unsqueeze(0), t_sample.unsqueeze(0), trunc=1e+9)


            if level > 0 and register_args['w_reg'] > 0:
                nonrigidity = data[level][1]
                target = torch.zeros_like(nonrigidity)
                reg_loss = BCE(nonrigidity, target)
                loss = loss + register_args['w_reg'] * reg_loss


            # early stop
            if loss.item() < 1e-4:
                break
            if abs(loss_prev - loss.item()) < loss_prev * register_args['break_threshold_ratio']:
                break_counter += 1
            if break_counter >= register_args['max_break_count']:
                break
            loss_prev = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # use warped points for next level
        s_sample = s_sample_warped.detach()
    model.gradient_setup(optimized_level=-1)
    src_tensor_transformed, data = model.warp(src_tensor_centered)
    src_tensor_transformed = src_tensor_transformed + tgt_tensor_mean
    if debug:
        src_pcd_arr = src_tensor.detach().cpu().numpy()
        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(src_pcd_arr)
        src_pcd_tf_arr = src_tensor_transformed.detach().cpu().numpy()
        src_transformed_pcd = o3d.geometry.PointCloud()
        src_transformed_pcd.points = o3d.utility.Vector3dVector(src_pcd_tf_arr)
        tgt_pcd_arr = tgt_tensor.detach().cpu().numpy()
        tgt_pcd = o3d.geometry.PointCloud()
        tgt_pcd.points = o3d.utility.Vector3dVector(tgt_pcd_arr)
        draw_registration_result(src_pcd, tgt_pcd, np.eye(4))
        draw_registration_result(src_transformed_pcd, tgt_pcd, np.eye(4))
    return src_tensor, src_tensor_transformed

def tacchifile_load(file:str):
    data = np.load(file)
    xi = data['p_xpos']
    yi = data['p_ypos']
    zi = data['p_zpos'].flatten()
    return np.stack((xi,yi,zi),axis=1)


if __name__ == "__main__":
    args = options()
    config = yaml.load(open(args.config,'r'), yaml.SafeLoader)
    device = config['DEVICE']
    model_args = config['model_args']
    register_args = config['register_args']
    resfresh_dir(args.output_dir)
    files = sorted(os.listdir(args.height_dir))
    src_pcd_arr = tacchifile_load(os.path.join(args.height_dir, files[0]))
    src_tensor = torch.from_numpy(src_pcd_arr).float().to(device)
    src_tensor_tf = src_tensor.clone()
    for file in tqdm(files[1:]):
        tgt_pcd = tacchifile_load(os.path.join(args.height_dir, file))
        tgt_tensor = torch.from_numpy(tgt_pcd).float().to(device)
        _, src_tensor_tf = NDP_register(model_args, register_args, src_tensor_tf, tgt_tensor, debug=True)
        flow = (src_tensor_tf - src_tensor).cpu().detach().numpy()
        np.save(os.path.join(args.output_dir,'flow_{}_{}.npy'.format(os.path.splitext(files[0])[0],os.path.splitext(file)[0])), flow)
    src_pcd_tf = o3d.geometry.PointCloud()
    src_pcd_tf_arr = src_tensor_tf.cpu().detach().numpy()
    src_pcd_tf.points = o3d.utility.Vector3dVector(src_pcd_tf_arr)
    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd_arr = tgt_tensor.cpu().detach().numpy()
    tgt_pcd.points = o3d.utility.Vector3dVector(tgt_pcd_arr)
    draw_registration_result(src_pcd_tf, tgt_pcd, np.eye(4))

    