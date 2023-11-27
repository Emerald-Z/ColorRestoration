import cv2
import numpy as np
import torch
import os
from recoloringnet import RecoloringNet, SimpleWithSkips

def yuv_to_img(y, uv, path="reconstructed_image", dim=300):
    yuv = np.zeros((dim, dim, 3), dtype=np.uint8)
    print(uv.shape)
    yuv[:,:,0] = y * 255
    yuv[:,:,1] = uv[0] * 255
    yuv[:,:,2] = uv[1] * 255

    img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    cv2.imwrite(f"{path}.jpg", img)

def test_with_mode():
    #change these or you could pass them in
    i = 99
    dim = 256
    model = SimpleWithSkips()
    y_base_dir = "./../IP_projest/val_256gray"
    uv_base_dir = "./../IP_projest/val_256color"
    model_name = "./../IP_projest/model_epoch_100_last"

    model.load_state_dict(torch.load(model_name))
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
    out = model(y).squeeze().detach().numpy() 
    y = y.squeeze().detach().numpy()
    yuv_to_img(y, out, path="model_out", dim=dim) #get the model out

if __name__=="__main__":
    test_with_mode()