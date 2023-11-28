from dataset import TacchiDataset
import taichi as ti
import numpy as np
import os
# import sys
import argparse

def options():
    parser = argparse.ArgumentParser()
    io_parser = parser.add_argument_group()
    io_parser.add_argument("--dataset_dir", type=str, default="tacchi_obj/obj_100")
    io_parser.add_argument("--output_dir", type=str, default="results_tr/posmap")
    io_parser.add_argument("--depth_file",type=str,default="results_tr/depth.npy")
    io_parser.add_argument("--scale_file",type=str,default="results_tr/scale.npy")
    io_parser.add_argument("--rot_file",type=str,default="results_tr/rot.npy")
    io_parser.add_argument("--index", type=int, default=0)
    io_parser.add_argument("--real_w",type=float,default=11.952,help="heigh of FOV image, unit:mm")
    io_parser.add_argument("--real_l",type=float,default=15.936,help="length of FOV image, unit:mm")
    io_parser.add_argument("--mmpp",type=float, default=0.0249, help="mm per pixel")
    run_parser = parser.add_argument_group()
    run_parser.add_argument("--max_height",type=str,default=8)
    run_parser.add_argument("--num_l", type=int, default=100)
    run_parser.add_argument("--num_w", type=int, default=100)
    run_parser.add_argument("--num_h",type=int,default=20)
    run_parser.add_argument("--scale_adjust",type=float,default=0.8)
    run_parser.add_argument("--x", type=int, default=0)
    run_parser.add_argument("--y", type=int, default=0)
    run_parser.add_argument("--gui",action="store_true",default=False)
    return parser.parse_args()



def get_last_layer_pose():
    "get the positions of the last layer of particles"
    x_ = x.to_numpy()
    p_xpos = x_[num_l*num_w*(num_h-1):num_l*num_w*num_h, 0]
    p_ypos = x_[num_l*num_w*(num_h-1):num_l*num_w*num_h, 1]
    p_zpos = x_[num_l*num_w*(num_h-1):num_l*num_w*num_h, 2]
    return p_xpos, p_ypos, p_zpos

@ti.kernel
def initialize(data: ti.types.ndarray(), data_len: ti.i32, ind_x: ti.f32, ind_y: ti.f32): # type: ignore
    for i, j, k in ti.ndrange(num_l, num_w, num_h):
        m = i+j*num_l+k*num_l*num_w
        offest = ti.Vector([l/2, w/2, 10-h/2])
        x[m] = ti.Vector([i, j, k])*dis+offest
        x_2d[m] = [x[m][0], x[m][1]]
        v[m] = [0, 0, 0]
        material[m] = 0
        F[m] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        C[m] = ti.Matrix.zero(float, 3, 3)

    for i in ti.ndrange(data_len):
        m = i+num_l*num_w*num_h
        offest = ti.Vector([20-ind_y, 20-ind_x, 10-h/2+args.max_height])
        x[m] = ti.Vector([-data[i, 0], -data[i, 1], -data[i, 2]])+offest  # type: ignore # (add particles in the center, reverse head)
        x_2d[m] = [x[m][0], x[m][1]]
        v[m] = [0, 0, 0]
        material[m] = 1
        F[m] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        C[m] = ti.Matrix.zero(float, 3, 3)

@ti.kernel
def substep():
    for i, j, k in grid_m:
        grid_v[i, j, k] = [0, 0, 0]
        grid_m[i, j, k] = 0

    # particle to grid
    for p in x:

        # first for particle p, compute base index
        base = (x[p] * inv_dx - 0.5).cast(int)

        # quadratic kernels
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1)
            ** 2, 0.5 * (fx - 0.5) ** 2]
        mu, la = mu_0, lambda_0
        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(3)):
            J *= sig[d, d]

        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + \
            ti.Matrix.identity(float, 3) * la * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]

        # P2G for velocity and mass
        # Loop over 3x3x3 grid node neighborhood
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1] * w[k][2]
            grid_m[base + offset] += weight * p_mass  # mass transfer
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)

    # grid operation
    for i, j, k in grid_m:
        if grid_m[i, j, k] > 0:
            grid_v[i, j, k] = (1 / grid_m[i, j, k]) * \
                grid_v[i, j, k]  # momentum to velocity

            # wall collisions - handle all 3 dimensions
            if i < 3 and grid_v[i, j, k][0] < 0:
                grid_v[i, j, k][0] = 0  # Boundary conditions
            if i > n_grid - 3 and grid_v[i, j, k][0] > 0:
                grid_v[i, j, k][0] = 0
            if j < 3 and grid_v[i, j, k][1] < 0:
                grid_v[i, j, k][1] = 0
            if j > n_grid - 3 and grid_v[i, j, k][1] > 0:
                grid_v[i, j, k][1] = 0
            if k < 3 and grid_v[i, j, k][2] < 0:
                grid_v[i, j, k][2] = 0
            if k > n_grid - 3 and grid_v[i, j, k][2] > 0:
                grid_v[i, j, k][2] = 0

    # grid to particle
    for p in x:

        # compute base index
        base = (x[p] * inv_dx - 0.5).cast(int)

        # quadratic kernels
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0)
            ** 2, 0.5 * (fx - 0.5) ** 2]

        new_v = ti.Vector.zero(float, 3)
        new_C = ti.Matrix.zero(float, 3, 3)
        # new_F = ti.Matrix.zero(float, 3, 3)

        # loop over 3x3x3 grid node neighborhood
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            dpos = ti.Vector([i, j, k]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j, k])]
            weight = w[i][0] * w[j][1] * w[k][2]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)

        # particle operation
        if p < num_l*num_w*3:
            new_v = ti.Vector([0, 0, 0])
        if material[p] == 1:
            new_C = ti.Matrix.zero(float, 3, 3)
            new_v = ti.Vector([0, 0, -200])

        v[p], C[p] = new_v, new_C

        # move the particles
        x[p] += dt * v[p]
        x_2d[p] = [x[p][1]*3-10, x[p][2]*3-10]  # update 2d positions
        F[p] = (ti.Matrix.identity(float, 3) + (dt * new_C)
                ) @ F[p]  # update F (explicitMPM way)
        
if __name__ == "__main__":
    args = options()
    if not os.path.isdir(args.output_dir):
        raise RuntimeError("{} should be created beforehand.".format(args.output_dir))
    # os.makedirs(args.output_dir,exist_ok=True)
    
    # taichi initialization
    ti.init(arch=ti.gpu)
    dt = 1e-4

    # gel and indenter initialization
    num_w = args.num_w + 1 # #intervals + 1 = #nodes
    num_l = args.num_l + 1
    num_h = args.num_h + 1

    l = 20
    w = 20
    h = 4

    dis = l/(num_l-1)

    dataset = TacchiDataset(args.dataset_dir)
    depth_list = np.load(args.depth_file)
    rot_list = np.load(args.rot_file)
    scale_list = np.load(args.scale_file)
    # world initialization
    t_ti = ti.field(dtype=ti.f32, shape=())
    t_ti[None] = 0
    
    # grid initialization
    n_grid = 256
    dx = 33 / n_grid
    inv_dx = 1 / dx
    
    grid_v = ti.Vector.field(3, dtype=float, shape=(
        n_grid, n_grid, n_grid))  # grid node momentum/velocity
    grid_m = ti.field(dtype=float, shape=(
        n_grid, n_grid, n_grid))  # grid node mass
    data = dataset.getitem(args.index % len(dataset), rot_list[args.index], args.scale_adjust*scale_list[args.index])
    max_depth = depth_list[args.index]
    data = data.astype(np.float32)
    # particle initialization
    n_particles = num_l*num_w*num_h+np.shape(data)[0]
    x = ti.Vector.field(3, dtype=float, shape=n_particles)  # position
    # 2d positions - this is necessary for circle visualization
    x_2d = ti.Vector.field(2, dtype=float, shape=n_particles)
    v = ti.Vector.field(3, dtype=float, shape=n_particles)  # velocity
    # affine velocity field
    C = ti.Matrix.field(3, 3, dtype=float, shape=n_particles)
    # deformation gradient
    F = ti.Matrix.field(3, 3, dtype=float, shape=n_particles)
    material = ti.field(dtype=int, shape=n_particles)  # material id


    p_vol, p_rho = (dx * 0.5)**2, 1
    p_mass = p_vol * p_rho
    E, nu = 1.45e5, 0.45  # Young's modulus and Poisson's ratio
    # Lame parameters - may change these later to model other materials
    mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1+nu) * (1 - 2 * nu))

    # initialization
    initialize(data, np.shape(data)[0], args.x, args.y)
    init_p_xpos_list, init_p_ypos_list, init_p_zpos_list = get_last_layer_pose()
    init_z_min = np.min(init_p_zpos_list)
    if args.gui:
        gui = ti.GUI("Explicit MPM rotate", res=400, background_color=0x112F41)
        colors = np.array([0x808080, 0x00ff00, 0xEEEEF0], dtype=np.uint32)
    # step sim
    while True:
        t_ti[None] += 1
        p_xpos_list, p_ypos_list, p_zpos_list = get_last_layer_pose()
        depth = np.min(p_zpos_list)  # 12 for intialization
        substep()
        if (depth-init_z_min) < 0.01 - max_depth:
            delta_xpos_list = (p_xpos_list - init_p_xpos_list).reshape(num_l, num_w).astype(np.float32)
            delta_ypos_list = (p_ypos_list - init_p_ypos_list).reshape(num_l, num_w).astype(np.float32)
            delta_zpos_list = (p_zpos_list - init_p_zpos_list).reshape(num_l, num_w).astype(np.float32)
            np.savez(os.path.join(args.output_dir, "{:04d}.npz".format(args.index)), \
                p_xpos_list=np.flipud(np.fliplr(-delta_xpos_list)),
                p_ypos_list=np.flipud(np.fliplr(-delta_ypos_list)),
                p_zpos_list=np.flipud(np.fliplr(-delta_zpos_list)))  # flip the gelsight image and inverte delta (because the reverse of x and y direction)
            break
        if args.gui:
            gui.circles(x_2d.to_numpy()/100, radius=1, color=colors[material.to_numpy()])
            gui.show()
    print("Index:{} steps: {} particles:{}".format(args.index, t_ti[None], data.shape[0]))
        
