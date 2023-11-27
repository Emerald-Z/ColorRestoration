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

class SimpleWithSkips(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(4,4), stride=2, padding=(1,1)) #one input channel
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4,4), stride=2, padding=(1,1))
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(4,4), stride=2, padding=(1,1))
        self.conv3_bn = nn.BatchNorm2d(128)

        self.t_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=(4,4), stride=2, padding=(1,1))
        self.t_conv1_bn = nn.BatchNorm2d(64)
        self.t_conv2 = nn.ConvTranspose2d(128, 32, kernel_size=(4,4), stride=2, padding=(1,1))
        self.t_conv2_bn = nn.BatchNorm2d(32)
        self.t_conv3 = nn.ConvTranspose2d(64, 2, kernel_size=(4,4), stride=2, padding=(1,1))

        self.output = nn.Conv2d(3, 2, kernel_size=(3,3), stride=1, padding=(1,1)) # two output channels
        
    def forward(self, x):
        x_1 = torch.relu(self.conv1_bn(self.conv1(x)))
        x_2 = torch.relu(self.conv2_bn(self.conv2(x_1)))
        x_3 = torch.relu(self.conv3_bn(self.conv3(x_2)))

        x_4 = torch.relu(self.t_conv1_bn(self.t_conv1(x_3)))
        x_4 = torch.cat((x_4, x_2), 1)
        x_5 = torch.relu(self.t_conv2_bn(self.t_conv2(x_4)))
        x_5 = torch.cat((x_5, x_1), 1)
        x_6 = torch.relu(self.t_conv3(x_5))
        x_6 = torch.cat((x_6, x), 1)
        out = self.output(x_6)
        return out
    
class SimpleRecoloringNet(nn.Module):
    def __init__(self):
        super(SimpleRecoloringNet, self).__init__()

      #future modifications
        # batch norm
        #
        self.encoder = torch.nn.Sequential(
          nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0),  # Input channels=1 for luminance value
          nn.ReLU(),
          nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
          nn.ReLU(),
          nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
          nn.ReLU()
        )
        self.decoder = nn.Sequential(
          nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=0),  # Reverse of encoder to end up with same dims
          nn.ReLU(),
          nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=0),
          nn.ReLU(),
          nn.ConvTranspose2d(16, 2, kernel_size=3, stride=1, padding=0, output_padding=0), #output channel = 2 for u/v color values
          nn.Sigmoid()  # scale pixel 0-1 #or we probably want to predict residuals like the difference in each pixel?
        )

    def forward(self, x):
      x = self.encoder(x)
      x = self.decoder(x)
      return x