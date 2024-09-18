import imageio
from PIL import Image
import os
path = "res/3d_reg"
output_file = 'gif/3d_reg.gif'

files = sorted(os.listdir(path))
gif_imgs = []
for file in files:
    img = Image.open(os.path.join(path, file)).convert('RGB')
    gif_imgs.append(img)
for _ in range(10):
    gif_imgs.append(img)  # wait
imageio.mimsave(output_file, gif_imgs, format='GIF', fps=4.0, loop=0)