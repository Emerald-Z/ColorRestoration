import torch
import torch.nn as nn
import torchvision.models as models

class RecoloringNet(nn.Module):
    def __init__(self):
        super(RecoloringNet, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(scale_factor=(2,2))
        self.conv2d_1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(3,3),stride=2,padding=(1,1))
        self.conv2d_2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=1,padding=(1,1))
        self.conv2d_3 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(3,3),stride=2,padding=(1,1))
        self.conv2d_4 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=1,padding=(1,1))
        self.conv2d_5 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(3,3),stride=2,padding=(1,1))
        self.conv2d_6 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(3,3),stride=1,padding=(1,1))
        self.conv2d_7 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3,3),stride=1,padding=(1,1))
        self.conv2d_8 = nn.Conv2d(in_channels=512,out_channels=256,kernel_size=(3,3),stride=1,padding=(1,1))

        self.conv2d_9 = nn.Conv2d(in_channels=1256,out_channels=256,kernel_size=(1,1),stride=1,padding=(0,0))

        self.conv2d_10 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=(3,3),stride=1,padding=(1,1))
        self.conv2d_11 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=(3,3),stride=1,padding=(1,1))
        self.conv2d_12 = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=(3,3),stride=1,padding=(1,1))
        self.conv2d_13 = nn.Conv2d(in_channels=32,out_channels=16,kernel_size=(3,3),stride=1,padding=(1,1))
        self.conv2d_14 = nn.Conv2d(in_channels=16,out_channels=2,kernel_size=(3,3),stride=1,padding=(1,1))

    def encoder(self, encoder_input):
        encoder_output = self.relu(self.conv2d_1(encoder_input))
        encoder_output = self.relu(self.conv2d_2(encoder_output))
        encoder_output = self.relu(self.conv2d_3(encoder_output))
        encoder_output = self.relu(self.conv2d_4(encoder_output))
        encoder_output = self.relu(self.conv2d_5(encoder_output))
        encoder_output = self.relu(self.conv2d_6(encoder_output))
        encoder_output = self.relu(self.conv2d_7(encoder_output))
        encoder_output = self.relu(self.conv2d_8(encoder_output))
        return encoder_output

    def decoder(self, decoder_input):
        decoder_output = self.relu(self.conv2d_10(decoder_input))
        decoder_output = self.upsample(decoder_output)
        decoder_output = self.relu(self.conv2d_11(decoder_output))
        decoder_output = self.upsample(decoder_output)
        decoder_output = self.relu(self.conv2d_12(decoder_output))
        decoder_output = self.relu(self.conv2d_13(decoder_output))
        decoder_output = self.sigmoid(self.conv2d_14(decoder_output))
        decoder_output = self.upsample(decoder_output)
        return decoder_output

    def fusion(self, embed_input, encoder_output):
        fusion_output = embed_input.reshape([-1,1000,1,1])
        fusion_output = fusion_output.repeat(1,1,32*32,1)
        fusion_output = torch.reshape(fusion_output, (-1,1000, 32,32))
        fusion_output = torch.cat((encoder_output, fusion_output), 1)
        fusion_output = self.relu(self.conv2d_9(fusion_output))
        return fusion_output

    def forward(self, x, embed_input):
      return self.decoder(self.fusion(embed_input, self.encoder(x)))

def load_model():
  """load the classifier, use eval as the classifier is not being trained during the model training"""
  inception = models.mobilenet_v2(pretrained=True)
  inception.eval()

  return inception