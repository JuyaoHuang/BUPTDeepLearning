import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28,512),
            nn.LeakyReLU(0.02),
            nn.Linear(512,128),
            nn.LeakyReLU(0.02),
            nn.Linear(128,64),
            nn.LeakyReLU(0.02),
            nn.Linear(64,10),
            nn.LeakyReLU(0.02),
            nn.Linear(10,2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(2,10),
            nn.LeakyReLU(0.02),
            nn.Linear(10, 64),
            nn.LeakyReLU(0.02),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.02),
            nn.Linear(128, 512),
            nn.LeakyReLU(0.02),
            nn.Linear(512, 28*28),
            nn.Sigmoid() # 归一化到 (0,1)
        )

    def forward(self, x):
        # x: [batch, 1, 28, 28]
        encode = self.encoder(x)
        decode = self.decoder(encode)
        decode = decode.view(x.size(0), 1, 28, 28)
        return encode, decode


if __name__ == '__main__':
    model = AutoEncoder()
    print(model)