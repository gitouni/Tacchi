import torch
import torch.nn as nn
BCE = nn.BCELoss()
import open3d as o3d

import torch.optim as optim

from tqdm import tqdm
import copy
from easydict import EasyDict as edict

from data_utils import setup_seed

import numpy as np

from ndp.nets import Deformation_Pyramid
from ndp.loss import compute_truncated_chamfer_distance
import argparse


setup_seed(0)

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

def pcd_loader(path:str, num_sampling:int):
    if path.endswith("pcd"):
        pcd = o3d.io.read_point_cloud(path)
    elif path.endswith("ply"):
        mesh = o3d.io.read_triangle_mesh(path)
        pcd = mesh.sample_points_uniformly(number_of_points=num_sampling)
    elif path.endswith("bin"):
        pcd_arr = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        pcd_arr = pcd_arr[pcd_arr[:,0]>0, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_arr)
        pcd = pcd.voxel_down_sample(0.15)
    else:
        raise NotImplementedError('Unknown file format. path:{}'.format(path))
    return pcd

def tacchifile_load(file:str):
    data = np.load(file)
    xi = data['p_xpos']
    yi = data['p_ypos']
    zi = data['p_zpos'].flatten()
    pcd = o3d.geometry.PointCloud()
    pcd_arr = np.stack((xi,yi,zi),axis=1)
    pcd.points = o3d.utility.Vector3dVector(pcd_arr)
    return pcd

if __name__ == "__main__":

    config = {
        "gpu_mode": True,

        "iters": 500,
        "lr": 0.01,
        "max_break_count": 15,
        "break_threshold_ratio": 0.001,

        "samples": 6000,

        "motion_type": "SE3",
        "rotation_format": "euler",

        "m": 9,
        "k0": -8,
        "depth": 3,
        "width": 128,
        "act_fn": "relu",

        "w_reg": 0,
        "w_ldmk": 0,
        "w_cd": 0.1
    }

    config = edict(config)

    if config.gpu_mode:
        config.device = "cuda:{}".format(torch.cuda.current_device())
    else:
        config.device = torch.device('cpu')


    parser = argparse.ArgumentParser()
    parser.add_argument('--source_file', type=str, default="results/0090.npz")
    parser.add_argument('--target_file', type=str, default="results/0100.npz")
    args = parser.parse_args()


    src_pcd_file = args.source_file
    tgt_pcd_file = args.target_file
    src_pcd = tacchifile_load(args.source_file)
    tgt_pcd = tacchifile_load(args.target_file)
    draw_registration_result(src_pcd, tgt_pcd, np.eye(4))

    """load data"""
    src_tensor:torch.Tensor = torch.from_numpy(np.array(src_pcd.points)).float().to(config.device) # (N, 3)
    tgt_tensor:torch.Tensor = torch.from_numpy(np.array(tgt_pcd.points)).float().to(config.device) # (N, 3)

    """construct model"""
    NDP = Deformation_Pyramid(depth=config.depth,
                              width=config.width,
                              device=config.device,
                              k0=config.k0,
                              m=config.m,
                              nonrigidity_est=config.w_reg > 0,
                              rotation_format=config.rotation_format,
                              motion=config.motion_type)



    """cancel global translation"""
    src_tensor_mean = src_tensor.mean(dim=0, keepdims=True)  # (N, 1)
    tgt_tensor_mean = tgt_tensor.mean(dim=0, keepdims=True)  # (N, 1)
    src_tensor = src_tensor - src_tensor_mean
    tgt_tensor = tgt_tensor - tgt_tensor_mean




    s_sample = src_tensor.clone()
    t_sample = tgt_tensor.clone()

    NDP.reset()
    for level in range(NDP.n_hierarchy):

        """freeze non-optimized level"""
        NDP.gradient_setup(optimized_level=level)

        optimizer = optim.Adam(NDP.pyramid[level].parameters(), lr=config.lr)

        break_counter = 0
        loss_prev = 1e+6

        """optimize current level"""
        for iter in tqdm(range(config.iters), desc="Level {}|{}".format(level, NDP.n_hierarchy)):


            s_sample_warped, data = NDP.warp(s_sample, max_level=level, min_level=level)

            loss = compute_truncated_chamfer_distance(s_sample_warped[None], t_sample[None], trunc=1e+9)


            if level > 0 and config.w_reg > 0:
                nonrigidity = data[level][1]
                target = torch.zeros_like(nonrigidity)
                reg_loss = BCE(nonrigidity, target)
                loss = loss + config.w_reg * reg_loss


            # early stop
            if loss.item() < 1e-4:
                break
            if abs(loss_prev - loss.item()) < loss_prev * config.break_threshold_ratio:
                break_counter += 1
            if break_counter >= config.max_break_count:
                break
            loss_prev = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # use warped points for next level
        s_sample = s_sample_warped.detach()



    """warp-original mesh verttices"""
    NDP.gradient_setup(optimized_level=-1)
    src_tensor_transformed, data = NDP.warp(src_tensor)
    src_tensor_transformed = src_tensor_transformed + tgt_tensor_mean
    src_pcd_arr = src_tensor_transformed.detach().cpu().numpy()
    src_transformed_pcd = o3d.geometry.PointCloud()
    src_transformed_pcd.points = o3d.utility.Vector3dVector(src_pcd_arr)
    draw_registration_result(src_transformed_pcd, tgt_pcd, np.eye(4))

    """dump results"""
    # o3d.io.write_triangle_mesh("sim3_demo/things4D/" + sname + "-fit.ply", src_mesh)