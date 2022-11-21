# Tacchi

This repository contains the code for the paper Tacchi: A Pluggable and Fast Gel Deformation Simulator for Optical Tactile Sensors. The code can be used for connection simulation with MuJoCo and Gazebo, Tacchi image generation, image quality comparison, and Sim2Real tasks. All the datasets generated by this repository can be found in [Tacchi_dataset](https://drive.google.com/drive/folders/1i83U_u2WEcEt4axol884JlPwI7MEZ6BS?usp=sharing). The presentation video is available in [Tacchi_video](https://drive.google.com/file/d/1dUDufy1mBJjZrX9N1kLRwV8a8PRwvDlO/view?usp=sharing).

<img src="https://github.com/zixichen007115/Tacchi/blob/main/gif/Tacchi.gif" height="250px"> <img src="https://github.com/zixichen007115/Tacchi/blob/main/gif/MuJoCo.gif" height="250px"> <img src="https://github.com/zixichen007115/Tacchi/blob/main/gif/images.gif" height="250px"> 

## 0_stl2npy
This package generates object files for Tacchi. To generate objects with different particles,

`python stl2ply.py`

`python ply2pcd.py --particle [particle number]`

`python pcd2npy.py`

## 1_connection_simulation
This package connects MuJoCo and Gazebo with Tacchi for simulation. The objects generated by **0_stl2npy** should be put in this package.

## 2_Tacchi_image_generation
This package generates images with Tacchi. The objects generated by **0_stl2npy** should be put in this package.

`python cmd.py --particle [particle number]`

## 3_align_and_comparison
This package is from the repository [Gelsight Simulation](https://github.com/danfergo/gelsight_simulation). This package can align images and evaluate simulation images. The images form **2_Tacchi_image_generation** should be put in this package. To align images,  

`python -m experiments.preparation.align_per_object`

`python -m experiments.preparation.generate_sim`

To evaluate simulation images, 

`python -m experiments.preparation.loss_eval`

## 4_AA21
This package applied depth map smoothing method applied in [18](https://ieeexplore.ieee.org/abstract/document/9561122) to generate tactile images. To generate images,

`python data_loader_AA21.py`

## 5_Sim2Real
This package applied ResNet18 to classifier images and evaluate images with Sim2Real tasks. The images form real experiments, [Gelsight Simulation](https://github.com/danfergo/gelsight_simulation), and **2_Tacchi_image_generation** should be put in this package. To train a ResNet, 

`python main.py --mode train --img_kind [image kind] --model_name [model name]`

To test a classifier with real images, 

`python main.py --mode test --img_kind real --model_name [model name]`

---

Thank [Daniel Fernandes Gomes](https://github.com/danfergo) for Gelsight rendering method, alignment method and image evaluation code.
