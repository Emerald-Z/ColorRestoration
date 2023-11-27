from dataset import generate_dataset, ImgDataset
from recoloringnet import RecoloringNet
from utilities import yuv_to_img
import torchvision
from recoloringnet import load_model
import torch
import torch.nn as nn
import cv2
import numpy as np

def train(dataset, validation_dataset, model, epochs, lr=.002, checkpnt_path="model"):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 64)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    #we can do some fancy loss with l2 distance, SSIM or other stuff but for now maybe just pixel differences
    #https://arxiv.org/pdf/1511.08861.pdf good losses for image restoration

    for epoch in range(epochs):
        for batch_idx, (y, uv, rgb) in enumerate(dataloader):
            y = y.float()
            uv = uv.float()
            optimizer.zero_grad()
            
            # rgb = np.concatenate([y, uv], axis=1)

            rgb = torch.Tensor(np.array(rgb, dtype='uint8'))

            with torch.no_grad():
              embed = inception(rgb) # should be rgb
            out = model(y, embed)  # Forward
            loss = criterion(out, uv)
            loss.backward()
            optimizer.step()
            print(f"----- epoch {epoch}---------")
            print(f"train loss {loss}")

        if epoch%10 == 0:
            #eval
            print(f"val loss {eval(validation_dataloader, model, criterion)}")
            torch.save(model.state_dict(), f"{checkpnt_path}_epoch_{epoch}.pt")

def eval(validation_loader, model, criterion):

# Set the model to evaluation mode (disables dropout and batch normalization)
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for y, uv, rgb in validation_loader:
            y = y.float()
            uv = uv.float()

            rgb = torch.Tensor(np.array(rgb, dtype='uint8'))
            with torch.no_grad():
              embed = inception(rgb)
            outputs = model(y, embed)
            loss = criterion(outputs, uv)
            total_loss += loss.item() * y.size(0)
    average_loss = total_loss / len(validation_loader.dataset)
    return average_loss

def validation_eval():
  torchvision.datasets.CelebA(root="ex\vl", split="validation", download=True)


def test_reconstruction():
    y = np.load("gray/processed_000003.npy")
    uv_real = np.load("color/processed_000003.npy")
    img = cv2.imread("data/000003.jpg")
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv_to_img(y, uv_real, 256)


if __name__ == "__main__":
    # torchvision.datasets.CelebA(root="ex", split="train", download=True)

    model = RecoloringNet()
    # generate_dataset("ex/celeba/img_align_celeba", 1000, 256)

    dataset = ImgDataset("gray", "color", "rgb")
    # for (y, uv), i in zip(dataset, range(3)):
    #     print("gray, ", y)
    #     print(y.shape)
    #     print("color, ", uv)
    #     print(uv.shape)

    inception = load_model()
    train(dataset, dataset, model, 101)

    # test
    # test_reconstruction()
    # saved_model = torch.load("model_epoch_8")
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1)

    # y_norm, uv_norm = dataset.__getitem__(0) # was 1
    # yuv = torch.Tensor(np.concatenate([y_norm, uv_norm], axis=0)).unsqueeze(0)

    # with torch.no_grad():
    #     embed = inception(torch.Tensor(yuv))
    #     uv_out = model(torch.Tensor(y_norm).unsqueeze(0), embed)
    # print(uv_out * 255)
    # yuv_to_img(y_norm * 255, uv_norm * 255, path="dataset")
    # yuv_to_img(y_norm * 255, uv_out.detach().numpy()[0] * 255, path="model")
