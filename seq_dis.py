import argparse
import os
import shutil
import numpy as np
from matplotlib import pyplot as plt

def resfresh_dir(dirname:str):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname)

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height_dir",type=str,default="res/height")
    parser.add_argument("--flow_dir",type=str,default="res/flow")
    parser.add_argument("--gt_dis",type=str,default="res/gt_dis")
    parser.add_argument("--pred_dis",type=str,default="res/pred_dis")
    parser.add_argument("--mmpp",type=float,default=0.0249)
    parser.add_argument("--grid_size",type=int,default=8)
    parser.add_argument("--w_margin",type=int,default=20)
    parser.add_argument("--h_margin",type=int,default=20)
    return parser.parse_args()


def tacchifile_load(file:str):
    data = np.load(file)
    xi = data['p_xpos']
    yi = data['p_ypos']
    zi = data['p_zpos'].flatten()
    return np.stack((xi,yi,zi),axis=1)

if __name__ == "__main__":
    args = options()
    height_files = sorted(os.listdir(args.height_dir))
    flow_files = sorted(os.listdir(args.flow_dir))
    resfresh_dir(args.gt_dis)
    resfresh_dir(args.pred_dis)
    gt_disx_dir = os.path.join(args.gt_dis, 'xflow')
    gt_disy_dir = os.path.join(args.gt_dis, 'yflow')
    pred_disx_dir = os.path.join(args.pred_dis, 'xflow')
    pred_disy_dir = os.path.join(args.pred_dis, 'yflow')
    os.makedirs(gt_disx_dir)
    os.makedirs(gt_disy_dir)
    os.makedirs(pred_disx_dir)
    os.makedirs(pred_disy_dir)
    ref_pcd = tacchifile_load(os.path.join(args.height_dir, height_files[0]))
    cur_pcd = tacchifile_load(os.path.join(args.height_dir, height_files[-1]))
    flow = cur_pcd - ref_pcd
    args_xflow = dict(vmin=np.min(flow[:,0]), vmax=np.max(flow[:,0]))
    args_yflow = dict(vmin=np.min(flow[:,1]), vmax=np.max(flow[:,1]))
    H, W = 201, 201
    um = []
    vm = []
    for i in range(args.grid_size):
        um.append(args.w_margin + i / args.grid_size * W)
        vm.append(args.h_margin + i / args.grid_size * H)
    um = np.array(um, dtype=np.int32)
    vm = np.array(vm, dtype=np.int32)
    ui, vi = np.meshgrid(np.arange(W), np.arange(H), indexing='ij')
    ui = ui.flatten()
    vi = vi.flatten()
    for i, (height_file, flow_file) in enumerate(zip(height_files[1:], flow_files)):
        cur_pcd = tacchifile_load(os.path.join(args.height_dir, height_file))
        gt_flow = cur_pcd - ref_pcd
        gt_flow_img = np.zeros([H, W, 2])
        gt_flow_img[vi, ui] = gt_flow[:, :2]
        pred_flow = np.load(os.path.join(args.flow_dir, flow_file))
        pred_flow_img = np.zeros_like(gt_flow_img)
        pred_flow_img[vi, ui] = pred_flow[:, :2]
        plt.imsave(os.path.join(gt_disx_dir, "%04d.png"%i), gt_flow_img[...,0], cmap='viridis',**args_xflow)
        plt.imsave(os.path.join(gt_disy_dir, "%04d.png"%i), gt_flow_img[...,1], cmap='viridis',**args_yflow)
        plt.imsave(os.path.join(pred_disx_dir, "%04d.png"%i), pred_flow_img[...,0], cmap='viridis',**args_xflow)
        plt.imsave(os.path.join(pred_disy_dir, "%04d.png"%i), pred_flow_img[...,1], cmap='viridis',**args_yflow)