import cv2
import numpy as np

def yuv_to_img(y, uv, img_size=256, path="reconstructed_image"):
    yuv = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    yuv[:,:,0] = y
    yuv[:,:,1] = uv[0]
    yuv[:,:,2] = uv[1]
    img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    cv2.imwrite(f"{path}.jpg", img)