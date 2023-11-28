import cv2
import numpy as np
import torch
import os

class ImgDataset(torch.utils.data.Dataset):
  def __init__(self, gray, color, rgb):
    gray_paths = os.listdir(gray)
    color_paths = os.listdir(color)
    rgb_paths = os.listdir(rgb)
    self.paths = list(set(gray_paths) & set(color_paths) & set(rgb_paths))

    self.gray = gray
    self.color = color
    self.rgb = rgb

  def __getitem__(self, index):
    y = np.load(os.path.join(self.gray, self.paths[index])) / 255. #normalize
    uv = np.load(os.path.join(self.color, self.paths[index])) / 255
    rgb = np.load(os.path.join(self.rgb, self.paths[index])) / 255
    return y, uv, rgb
  
  def __len__(self):
    return len(self.paths)

def generate_dataset(input_path, max_num, img_size):
    for filename, i in zip(os.listdir(input_path), range(max_num)):
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

        img = cv2.resize(gray_img, (w, h))
        filename = os.path.splitext(filename)[0]
        rgb_image = np.transpose(cv2.resize(img, (img_size, img_size)), (2, 0, 1))
        output_path = os.path.join("rgb", f"processed_{filename}")
        np.save(output_path, np.array(rgb_image, dtype='uint8'))

        start_x = max(w // 2 - img_size // 2, 0)
        start_y = max(h // 2 - img_size // 2, 0)
        img = img[start_y:start_y + img_size, start_x:start_x + img_size]
        output_path = os.path.join("color", f"processed_{filename}.npy")
        yuv_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        Y = yuv_image[:, :, 0]
        U = yuv_image[:, :, 1]
        V = yuv_image[:, :, 2]
        # np.save(output_path, np.stack((U, V)))
        output_path = os.path.join("gray", f"processed_{filename}")
        # np.save(output_path, Y.reshape(1, img_size, img_size))

