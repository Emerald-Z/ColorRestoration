import cv2
import numpy as np
import torch
import os
from recoloringnet import RecoloringNet, SimpleWithSkips, BetterWithSkips, SkipConnectionUnet, SimpleRecoloringNet

def yuv_to_img(y, uv, path="reconstructed_image", dim=300, bgr=True):
    yuv = np.zeros((dim, dim, 3), dtype=np.uint8)
    print(uv.shape)
    yuv[:,:,0] = y * 255
    yuv[:,:,1] = uv[0] * 255
    yuv[:,:,2] = uv[1] * 255

    if bgr:
        img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    else:
        img = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    cv2.imwrite(f"{path}.jpg", img)

def test_with_model():
    #change these or you could pass them in
    i = 330
    dim = 256
    model = SimpleWithSkips()
    y_base_dir = "./val_gray"
    uv_base_dir = "./val_color"
    model_name = "./model_epoch_90.pt"

    model.load_state_dict(torch.load(model_name))
    paths = os.listdir(y_base_dir)
    y = np.load(f"{y_base_dir}/{paths[i]}") / 255.
    uv = np.load(f"{uv_base_dir}/{paths[i]}") / 255.
    yuv_to_img(y, uv, dim=dim) #get the original 
    y = torch.tensor(y.reshape(1, 1, dim, dim))
    y = y.float()
    print(y)
    # uv = torch.ones(1, 2, dim, dim) * 0
    # uv = uv.float()
    # yuv = torch.Tensor(np.concatenate([y, uv], axis=1))
    # embed = inception(yuv)
    out = model(y).squeeze().detach().numpy() 
    y = y.squeeze().detach().numpy()
    yuv_to_img(y, out, path="model_out", dim=dim, bgr=True) #get the model out

def test_with_model64():
    #change these or you could pass them in
    i = 1
    dim =256
    model = SimpleRecoloringNet()
    prefix = "t_"
    #dataset = ImgDataset(prefix + "train_gray", prefix + "train_color")
    #val_dataset = ImgDataset(prefix + "val_gray", prefix + "val_color")
    y_base_dir = prefix + "gray"
    uv_base_dir = prefix + "color"
    model_name = "./model_epoch_295-basic.pt"
    model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))
    paths = os.listdir(y_base_dir)
    y = np.load(f"{y_base_dir}/{paths[i]}") / 255.
    uv = np.load(f"{uv_base_dir}/{paths[i]}") / 255.
    yuv_to_img(y, uv, dim=dim) #get the original 
    y = torch.tensor(y.reshape(1, 1, dim, dim))
    y = y.float()
    # uv = torch.ones(1, 2, dim, dim) * 0
    # uv = uv.float()
    # yuv = torch.Tensor(np.concatenate([y, uv], axis=1))
    # embed = inception(yuv)
    y = y.reshape(dim, dim, 1).detach().numpy()
    input = torch.Tensor(cv2.resize(y, (64, 64)).reshape(1, 1, 64, 64))
    out = model(input).squeeze().detach().numpy() 
    out = out.reshape(64, 64, 2)
    out = cv2.resize(out, (dim, dim))
    out = out.reshape(2, dim, dim)
    y = y.squeeze()
    yuv_to_img(y, out, path="model_out", dim=dim, bgr=True) #get the model out


def test_color_compression():
    i = 1000
    dim = 256
    prefix = "ia_"
    y_base_dir = prefix + "train_gray"
    uv_base_dir = prefix + "train_color"
    paths = os.listdir(y_base_dir)

    y = np.load(f"{y_base_dir}/{paths[i]}") / 255.
    uv = np.load(f"{uv_base_dir}/{paths[i]}") / 255.
    # print(uv)
    yuv_to_img(y, uv, dim=dim, path="normal_color1")
    uv = uv.reshape(dim, dim, 2)
    uv = cv2.resize(uv, (64, 64))
    uv = cv2.resize(uv, (dim, dim))
    uv = uv.reshape(2, dim, dim)
    # print(uv)
    yuv_to_img(y, uv, path="compressed_color1", dim=dim) #get the model out


if __name__=="__main__":
    test_with_model64()
    #test_color_compression()