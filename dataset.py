# import h5py
import os
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats.qmc import Halton
import open3d as o3d
import glob
# from typing import Literal

def eulerToMat(x:float, y:float, z:float):
    R = Rotation.from_euler('zyx',[z,y,x])
    return R.as_matrix()

def vecToMat(x:np.ndarray):
    X = np.eye(4)
    X[:3,:3] = eulerToMat(*x[:3])
    X[:3, 3] = x[3:]
    return X

class AbstractDataset:
    def __init__(self, **kwargs):
        pass
    
    def __len__(self) -> int:
        raise NotImplementedError("Cannot return __len__ of the Abstract Class")
    
    def __getitem__(self, index:int) -> np.ndarray:
        raise NotImplementedError("Abstract Dataset cannot be used directly.")
    
    def getitem(self, index:int, rotate:list, scale:float):
        data = self[index] * scale
        rotation_matrix = eulerToMat(*rotate)
        data = np.transpose(rotation_matrix @ data.T, (1,0))
        if len(data) != 0:
            data[:,2] -= np.min(data[:,2])
        return data

# class ModelNetH5Dataset(AbstractDataset):
#     def __init__(self, data_path:str, constant_scale=4):
#         assert os.path.isfile(data_path), "{} is not a valid file.".format(data_path)
#         self.h5pyfile = h5py.File(data_path , 'r')
#         self.constant_scale = constant_scale
        
#     def __len__(self):
#         return self.h5pyfile['points'].shape[0]
    
#     def __getitem__(self, index):
#         return self.constant_scale*self.h5pyfile['points'][index]
    
#     def getitem(self, index:int, rotate:list, scale:float):
#         return super().getitem(index, rotate, scale)
    
#     def __del__(self):
#         self.h5pyfile.close()

# class ModelNetDataset(AbstractDataset):
#     def __init__(self, data_path:str, phase:Literal['train','test'], skip:int=1, num_sample_pts:int=200000, constant_scale:float=16):
#         assert os.path.isdir(data_path), "{} is not a valid file.".format(data_path)
#         self.root = data_path
#         subdir_list = sorted(os.listdir(data_path))
#         self.subdir_list = [os.path.join(path, phase) for path in subdir_list]
#         self.num_file_list = []
#         self.subfile_list = []
#         for subdir in self.subdir_list:
#             subfiles = [file for file in os.listdir(os.path.join(data_path, subdir)) if os.path.splitext(file)[1] in ['.off','.ply','.obj']]
#             self.subfile_list.append(subfiles)
#             self.num_file_list.append(len(subfiles))
#         self.num_file_list = np.array(self.num_file_list, dtype=np.int32)
#         self.cum_file_list = np.cumsum(self.num_file_list)
#         self.skip = skip
#         self.num_sample = num_sample_pts
#         self.constant_scale = constant_scale
        
#     def __len__(self):
#         return self.num_file_list.sum() // self.skip
    
#     def __getitem__(self, index):
#         file_index = index * self.skip
#         subdir_index = np.digitize(file_index, self.cum_file_list)
#         subfile_index = file_index - self.cum_file_list[subdir_index]
#         filename = os.path.join(self.root, self.subdir_list[subdir_index], self.subfile_list[subdir_index][subfile_index])
#         mesh = o3d.io.read_triangle_mesh(filename)
#         vertices = np.array(mesh.vertices)
#         delta = np.amax(vertices, axis=0) - np.amin(vertices, axis=0)
#         mesh.scale(self.constant_scale / delta.max(), (0,0,0))
#         pt = mesh.sample_points_uniformly(self.num_sample)
#         return np.array(pt.points)
    
#     def getitem(self, index:int, rotate:list, scale:float):
#         return super().getitem(index, rotate, scale)
    

class TacchiDataset(AbstractDataset):
    def __init__(self, data_path:str) -> None:
        assert os.path.isdir(data_path), "{} is not a valid file.".format(data_path)
        self.data_path = data_path
        self.obj_name_list = list(sorted(os.listdir(data_path)))
    
    def __len__(self):
        return len(self.obj_name_list)
    
    @staticmethod
    def data_transfer(obj_name:str, data_:np.ndarray) -> np.ndarray:
        data = data_.copy()
        if obj_name == "cross_lines" or obj_name == "cylinder_side" or obj_name == "hexagon" or obj_name == "line":
            data[:,0]=data_[:,1]
            data[:,1]=data_[:,0]
        elif obj_name == "dots":
            data[:,1]=-data_[:,1]
        elif obj_name == "wave1":
            data[:,0]=-data_[:,0]
        elif obj_name == "moon" or obj_name == "pacman":
            data[:,0]=data_[:,1]
            data[:,1]=-data_[:,0]
        elif obj_name == "triangle":
            data[:,0]=-data_[:,1]
            data[:,1]=data_[:,0]
        return data
    
    def __getitem__(self, index:int) -> np.ndarray:
        data = np.load(os.path.join(self.data_path, self.obj_name_list[index]))
        name = os.path.splitext(self.obj_name_list[index])[0]
        data = self.data_transfer(name, data)
        return data
    
    def getitem(self, index:int, rotate:list, scale:float):
        return super().getitem(index, rotate, scale)

# class YCBDataset(AbstractDataset):
#     def __init__(self, data_path:str, subdir="clouds", filename="merged_cloud.ply", constant_scale=100.0):
#         self.data_path = data_path
#         self.subpath_list = sorted(os.listdir(data_path))
#         self.subpath_list = [path for path in self.subpath_list if os.path.isdir(os.path.join(data_path, path, 'clouds'))]
#         self.subdir = subdir
#         self.filename = filename
#         self.constant_scale = constant_scale
#     def __len__(self):
#         return len(self.subpath_list)
    
#     def __getitem__(self, index: int) -> np.ndarray:
#         tgt_filename = os.path.join(self.data_path, self.subpath_list[index], self.subdir, self.filename)
#         if os.path.isfile(tgt_filename):
#             pt_cloud = o3d.io.read_point_cloud(tgt_filename)
#         else:
#             tgt_filename = glob.glob(os.path.join(self.data_path, self.subpath_list[index], "**/*.obj"), recursive=True)[0]
#             mesh = o3d.io.read_triangle_mesh(tgt_filename)
#             pt_cloud = mesh.sample_points_uniformly(number_of_points=50000)
#         return np.array(pt_cloud.points) * self.constant_scale
    
#     def getitem(self, index: int, rotate: list, scale: float):
#         return super().getitem(index, rotate, scale)
    

class TacchiRandomDataset(TacchiDataset):
    def __init__(self, data_path:str, data_len:int, seeds=[7542,1647,2451], scale_range=[0.8,1.2], rot_amp=[0.1,0.1,2.0], depth_range=[0.8,2.5]):
        super().__init__(data_path)
        scale_sampler = Halton(1, seed=seeds[0])
        rot_sampler = Halton(3, seed=seeds[1])
        depth_sampler = Halton(1, seed=seeds[2])
        self.scale_data = scale_sampler.random(data_len) * (scale_range[1] - scale_range[0]) + scale_range[0]
        self.rot_data = 2 * (rot_sampler.random(data_len) - 0.5) * np.array(rot_amp)[None,:]
        self.depth_data = depth_sampler.random(data_len) *  (depth_range[1] - depth_range[0]) + depth_range[0]
    
    def getitem(self, index:int):
        data = super().getitem(index % super().__len__(), self.rot_data[index], self.scale_data[index])
        return data, self.depth_data[index]
    
    def __len__(self):
        return len(self.scale_data)
    
    
# if __name__ == "__main__":
#     data_path = os.path.join(os.path.dirname(__file__), "./modelnet/ModelNet40")
#     dataset = ModelNetDataset(os.path.abspath(data_path), 'train', skip=10)
#     print("total data: {}".format(len(dataset)))
#     data_ex = dataset[int(0.5*len(dataset))]
#     print("min:{}, max:{}".format(np.amin(data_ex, axis=0), np.amax(data_ex, axis=0)))