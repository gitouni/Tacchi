import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d


file = 'res/height/0070.npz'
data = np.load(file)
z:np.ndarray = data['p_zpos']
H, W = z.shape
ui, vi = np.meshgrid(np.arange(W), np.arange(H), indexing='ij')
ui = ui.flatten()
vi = vi.flatten()
zi = z.flatten()
depth = np.zeros([H, W])
depth[vi, ui] = zi
plt.imshow(depth)
plt.show()
xi = data['p_xpos']
yi = data['p_ypos']
pcd = o3d.geometry.PointCloud()
pcd_arr = np.stack((xi,yi,zi),axis=1)
pcd.points = o3d.utility.Vector3dVector(pcd_arr)
o3d.visualization.draw_geometries([pcd])