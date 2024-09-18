# modified from https://github.com/MCG-NJU/CamLiFlow/blob/main/utils.py

import re
import cv2
import numpy as np
import torch
# from typing import Literal
# from scipy.interpolate import griddata

# def img_closing(mask:np.ndarray, ksize=(7,7), iterations=2):
#     morphclose_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
#     return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morphclose_kernel, iterations=iterations)

# def simple_value_completion(img:np.ndarray, mask:np.ndarray, inplace=False, interpolate_mode:Literal['linear','cubic','nearest']='cubic'):
#     if not inplace:
#         new_img = img.copy()
#     else:
#         new_img = img
#     pts_with_value = np.nonzero(mask)
#     pts_without_value = np.nonzero(mask == 0)
#     fi = griddata(pts_with_value, img[pts_with_value[0], pts_with_value[1]], pts_without_value, method=interpolate_mode)
#     new_img[pts_without_value[0], pts_without_value[1]] = fi
#     return new_img

def setup_seed(seed):
    """
    fix random seed for deterministic training
    """
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def nptran(pcd:np.ndarray, G:np.ndarray, inplace=False) -> np.ndarray:
    """transform a np.ndarray point cloud (pcd) using G

    Args:
        pcd (np.ndarray): N,3
        G (np.ndarray): 4,4
        inplace (bool, optional): whether to inplace the raw point cloud. Defaults to False.

    Returns:
        np.ndarray: transformed point cloud
    """
    if not inplace:
        _pcd = np.copy(pcd)
    else:
        _pcd = pcd
    _pcd = np.dot(_pcd, G[:3,:3].T)
    _pcd += G[:3, [3]].T
    return _pcd


def npproj(pcd:np.ndarray, extran:np.ndarray, intran:np.ndarray, img_shape:tuple):
    """_summary_

    Args:
        pcd (np.ndarray): Nx3\\
        extran (np.ndarray): 4x4\\
        intran (np.ndarray): 3x3\\
        img_shape (tuple): HxW\\

    Returns:
        _type_: uv (N,2), rev (N,)
    """
    H, W = img_shape[0], img_shape[1]
    pcd_ = nptran(pcd, extran)  # (N, 3)
    if intran.shape[1] == 4:
        pcd_ = intran @ np.concatenate([pcd_, np.ones([pcd_.shape[0],1])],axis=1).T
    else:
        pcd_ = intran @ pcd_.T  # (3, N)
    u, v, w = pcd_[0], pcd_[1], pcd_[2]
    raw_index = np.arange(u.size)
    rev = w > 0
    raw_index = raw_index[rev]
    u = u[rev]/w[rev]
    v = v[rev]/w[rev]
    rev2 = (0<=u) * (u<W) * (0<=v) * (v<H)
    return np.stack((u[rev2],v[rev2]),axis=1), raw_index[rev2]  # (N, 2), (N,)


def npproj_wocons(pcd:np.ndarray, extran:np.ndarray, intran:np.ndarray):
    """_summary_

    Args:
        pcd (np.ndarray): Nx3\\
        extran (np.ndarray): 4x4\\
        intran (np.ndarray): 3x3\\

    Returns:
        _type_: uv (N,2), rev (N,)
    """
    pcd_ = nptran(pcd, extran)  # (N, 3)
    if intran.shape[1] == 4:
        pcd_ = intran @ np.concatenate([pcd_, np.ones([pcd_.shape[0],1])],axis=1).T
    else:
        pcd_ = intran @ pcd_.T  # (3, N)
    u, v, w = pcd_[0], pcd_[1], pcd_[2]
    u = u/w
    v = v/w
    return np.stack((u,v),axis=1)  # (N, 2), (N,)

def project_corr_pts(src_pcd_corr:np.ndarray, tgt_pcd_corr:np.ndarray, extran:np.ndarray, intran:np.ndarray, img_shape:tuple, toint32:bool=False, return_indices:bool=False):
    src_proj_pts, src_rev_idx = npproj(src_pcd_corr, extran, intran, img_shape)
    tgt_proj_pts = npproj_wocons(tgt_pcd_corr[src_rev_idx], extran, intran)
    if toint32:
        src_proj_pts = src_proj_pts.astype(np.int32)
        tgt_proj_pts = tgt_proj_pts.astype(np.int32)
    if return_indices:
        return src_proj_pts, tgt_proj_pts, src_rev_idx
    else:
        return src_proj_pts, tgt_proj_pts


def load_fpm(filename):
    with open(filename, 'rb') as f:
        header = f.readline().rstrip()
        if header.decode("ascii") == 'PF':
            color = True
        elif header.decode("ascii") == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception('Malformed PFM header.')

        scale = float(f.readline().decode("ascii").rstrip())
        if scale < 0:  # little-endian
            endian = '<'
        else:
            endian = '>'  # big-endian

        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)

    return data


def load_flow(filepath):
    with open(filepath, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert (202021.25 == magic), 'Invalid .flo file: incorrect magic number'
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        flow = np.fromfile(f, np.float32, count=2 * w * h).reshape([h, w, 2])

    return flow


def load_flow_png(filepath, scale=64.0):
    # for KITTI which uses 16bit PNG images
    # see 'https://github.com/ClementPinard/FlowNetPytorch/blob/master/datasets/KITTI.py'
    # The -1 is here to specify not to change the image depth (16bit), and is compatible
    # with both OpenCV2 and OpenCV3
    flow_img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    flow = flow_img[:, :, 2:0:-1].astype(np.float32)
    mask = flow_img[:, :, 0] > 0
    flow = flow - 32768.0
    flow = flow / scale
    return flow, mask


def save_flow(filepath, flow):
    assert flow.shape[2] == 2
    magic = np.array(202021.25, dtype=np.float32)
    h = np.array(flow.shape[0], dtype=np.int32)
    w = np.array(flow.shape[1], dtype=np.int32)
    with open(filepath, 'wb') as f:
        f.write(magic.tobytes())
        f.write(w.tobytes())
        f.write(h.tobytes())
        f.write(flow.tobytes())


def save_flow_png(filepath, flow, mask=None, scale=64.0):
    assert flow.shape[2] == 2
    assert np.abs(flow).max() < 32767.0 / scale
    flow = flow * scale
    flow = flow + 32768.0

    if mask is None:
        mask = np.ones_like(flow)[..., 0]
    else:
        mask = np.float32(mask > 0)

    flow_img = np.concatenate([
        mask[..., None],
        flow[..., 1:2],
        flow[..., 0:1]
    ], axis=-1).astype(np.uint16)

    cv2.imwrite(filepath, flow_img)


def load_disp_png(filepath):
    array = cv2.imread(filepath, -1)
    valid_mask = array > 0
    disp = array.astype(np.float32) / 256.0
    disp[np.logical_not(valid_mask)] = -1.0
    return disp, valid_mask


def save_disp_png(filepath, disp, mask=None):
    if mask is None:
        mask = disp > 0
    disp = np.uint16(disp * 256.0)
    disp[np.logical_not(mask)] = 0
    cv2.imwrite(filepath, disp)


def load_calib(filepath):
    with open(filepath) as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('P_rect_02'):
                proj_mat = line.split()[1:]
                proj_mat = [float(param) for param in proj_mat]
                proj_mat = np.array(proj_mat, dtype=np.float32).reshape(3, 4)
                assert proj_mat[0, 1] == proj_mat[1, 0] == 0
                assert proj_mat[2, 0] == proj_mat[2, 1] == 0
                assert proj_mat[0, 0] == proj_mat[1, 1]
                assert proj_mat[2, 2] == 1

    return proj_mat


def zero_padding(inputs, pad_h, pad_w):
    input_dim = len(inputs.shape)
    assert input_dim in [2, 3]

    if input_dim == 2:
        inputs = inputs[..., None]

    h, w, c = inputs.shape
    assert h <= pad_h and w <= pad_w

    result = np.zeros([pad_h, pad_w, c], dtype=inputs.dtype)
    result[:h, :w] = inputs

    if input_dim == 2:
        result = result[..., 0]

    return result


def disp2pc(disp, baseline, f, cx, cy, flow=None):
    h, w = disp.shape
    depth = baseline * f / (disp + 1e-5)

    xx = np.tile(np.arange(w, dtype=np.float32)[None, :], (h, 1))
    yy = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))

    if flow is None:
        x = (xx - cx) * depth / f
        y = (yy - cy) * depth / f
    else:
        x = (xx - cx + flow[..., 0]) * depth / f
        y = (yy - cy + flow[..., 1]) * depth / f

    pc = np.concatenate([
        x[:, :, None],
        y[:, :, None],
        depth[:, :, None],
    ], axis=-1)

    return pc


def project_pc2image(pc, image_h, image_w, f, cx=None, cy=None, clip=True):
    pc_x, pc_y, depth = pc[..., 0], pc[..., 1], pc[..., 2]

    cx = (image_w - 1) / 2 if cx is None else cx
    cy = (image_h - 1) / 2 if cy is None else cy

    image_x = cx + (f / depth) * pc_x
    image_y = cy + (f / depth) * pc_y

    if clip:
        return np.concatenate([
            np.clip(image_x[:, None], a_min=0, a_max=image_w - 1),
            np.clip(image_y[:, None], a_min=0, a_max=image_h - 1),
        ], axis=-1)
    else:
        return np.concatenate([
            image_x[:, None],
            image_y[:, None]
        ], axis=-1)