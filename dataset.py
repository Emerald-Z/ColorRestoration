import cv2
import numpy as np
import torch
import os
import random

class ImgDataset(torch.utils.data.Dataset):
  def __init__(self, gray, color, max):
    gray_paths = os.listdir(gray)
    color_paths = os.listdir(color)
    self.paths = list(set(gray_paths) & set(color_paths))[0:max]

    self.gray = gray
    self.color = color
    print(len(self.paths))

  def __getitem__(self, index):
    y = np.load(os.path.join(self.gray, self.paths[index])) / 255. #normalize
    uv = np.load(os.path.join(self.color, self.paths[index])) / 255
    y = y.squeeze()
    y = cv2.resize(y, (64, 64))
    uv = uv.reshape(256, 256, 2)
    uv = cv2.resize(uv, (64, 64))
    y = y.reshape(1, 64, 64)
    uv = uv.reshape(2, 64, 64)
    return y, uv
  
  def __len__(self):
    return len(self.paths)
  
class ImgDataset64(torch.utils.data.Dataset):
  def __init__(self, gray, color):
    gray_paths = os.listdir(gray)
    color_paths = os.listdir(color)
    self.paths = list(set(gray_paths) & set(color_paths))

    self.gray = gray
    self.color = color
    print(len(self.paths))

  def __getitem__(self, index):
    y = np.load(os.path.join(self.gray, self.paths[index])) / 255. #normalize
    uv = np.load(os.path.join(self.color, self.paths[index])) / 255
    y = y.squeeze()
    y = cv2.resize(y, (64, 64))
    uv = uv.reshape(256, 256, 2)
    uv = cv2.resize(uv, (64, 64))
    y = y.reshape(1, 64, 64)
    uv = uv.reshape(2, 64, 64)

    y = torch.tensor(y).to('cuda')
    uv = torch.tensor(uv).to('cuda')
    return y, uv
  
  def __len__(self):
    return len(self.paths)

def generate_dataset(input_path, max_num, img_size, path_base="", depth=1):
    paths = os.listdir(input_path)
    real_paths = []
    if (depth==2):
        for path in paths:
            if not path.startswith('.'):
                real_paths.extend([os.path.join(path, x) for x in os.listdir(os.path.join(input_path, path))])
    else: real_paths = paths
    random.shuffle(real_paths)
    for filename, i in zip(real_paths, range(max_num)):
        img_path = os.path.join(input_path, filename)
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        if height >= width:
            w = img_size
            h = int(height * (w / width))
        else:
            h = img_size
            w = int(width * (h / height))

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_img = cv2.merge([gray_img, gray_img, gray_img])

        gray_img = cv2.resize(gray_img, (w, h))
        img = cv2.resize(img, (w, h))
        filename = os.path.basename(filename)
        filename = os.path.splitext(filename)[0]
        rgb_image = np.transpose(cv2.resize(gray_img, (img_size, img_size)), (2, 0, 1))
        output_path = os.path.join(path_base + "rgb", f"processed_{filename}")
        np.save(output_path, np.array(rgb_image, dtype='uint8'))

        start_x = max(w // 2 - img_size // 2, 0)
        start_y = max(h // 2 - img_size // 2, 0)
        img = img[start_y:start_y + img_size, start_x:start_x + img_size]
        output_path = os.path.join(path_base + "color", f"processed_{filename}.npy")
        yuv_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        Y = yuv_image[:, :, 0]
        U = yuv_image[:, :, 1]
        V = yuv_image[:, :, 2]
        np.save(output_path, np.stack((U, V)))
        output_path = os.path.join(path_base + "gray", f"processed_{filename}")
        np.save(output_path, Y.reshape(1, img_size, img_size))

if __name__ == "__main__":
    generate_dataset("test2", max_num=5, img_size=256, path_base="t_", depth=1)