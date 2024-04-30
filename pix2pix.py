import torch
import torch.nn as nn
import torchvision.datasets as dataset

class Generator(nn.Module):
    def __init__(self, data_len=70, kernel_size=5, ngf=64, drop_rate=0.5):
        super(Generator, self).__init__()
        # Encoder
        # input: 70
        self.en1 = nn.Sequential(
            # nn.BatchNorm1d(1),
            nn.Conv1d(1, ngf, kernel_size=kernel_size, stride=1, padding=2),
            # 输入图片已正则化 不需BatchNorm
            nn.LeakyReLU(0.2, inplace=True),
        )
        # size: 70
        self.en2 = nn.Sequential(
            nn.Conv1d(ngf, ngf * 2, kernel_size=kernel_size, stride=2, padding=3),
            nn.BatchNorm1d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # size: 36
        self.en3 = nn.Sequential(
            nn.Conv1d(ngf * 2, ngf * 4, kernel_size=kernel_size, stride=2, padding=2),
            nn.BatchNorm1d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # size: 18
        self.en4 = nn.Sequential(
            nn.Conv1d(ngf * 4, ngf * 8, kernel_size=kernel_size, stride=2, padding=2),
            nn.BatchNorm1d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # size:  9
        self.en5 = nn.Sequential(
            nn.Conv1d(ngf * 8, ngf * 8, kernel_size=kernel_size, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True)
            # size:  4
        )


        # Decoder
        # input: 4
        self.de1 = nn.Sequential(
            nn.ConvTranspose1d(ngf * 8, ngf * 8, kernel_size=kernel_size, stride=2, padding=2),
            nn.BatchNorm1d(ngf * 8),
            nn.Dropout(drop_rate),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # size: 9
        self.de2 = nn.Sequential(
            nn.ConvTranspose1d(ngf * 8 * 2, ngf * 4, kernel_size=kernel_size, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(ngf * 4),
            nn.Dropout(drop_rate),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # size: 18
        self.de3 = nn.Sequential(
            nn.ConvTranspose1d(ngf * 4 * 2, ngf * 2, kernel_size=kernel_size, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(ngf * 2),
            nn.Dropout(drop_rate),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # size: 36
        self.de4 = nn.Sequential(
            nn.ConvTranspose1d(ngf * 2 * 2, ngf, kernel_size=kernel_size, stride=2, padding=3, output_padding=1),
            nn.BatchNorm1d(ngf),
            nn.Dropout(drop_rate),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # size: 70
        self.de5 = nn.Sequential(
            nn.ConvTranspose1d(ngf * 2, 1, kernel_size=5, stride=1, padding=2),
            nn.Tanh()
        )


    def forward(self, x):
        x = x.unsqueeze(1)
        # Encoder
        en1_out = self.en1(x)
        en2_out = self.en2(en1_out)
        en3_out = self.en3(en2_out)
        en4_out = self.en4(en3_out)
        en5_out = self.en5(en4_out)

        # Decoder
        de1_out = self.de1(en5_out)
        de1_cat = torch.cat([de1_out, en4_out], dim=1)  # cat by channel
        de2_out = self.de2(de1_cat)
        de2_cat = torch.cat([de2_out, en3_out], dim=1)
        de3_out = self.de3(de2_cat)
        de3_cat = torch.cat([de3_out, en2_out], dim=1)
        de4_out = self.de4(de3_cat)
        de4_cat = torch.cat([de4_out, en1_out], dim=1)
        out = self.de5(de4_cat)

        return out.squeeze(1)


class Discriminator(nn.Module):
    def __init__(self, data_len=70, kernel_size=5, ngf=64, drop_rate=0.25):
        super(Discriminator, self).__init__()
        # input: 70 * 2
        self.en1 = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Conv1d(1, ngf, kernel_size=kernel_size, stride=1, padding=2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # size: 70 * 2
        self.en2 = nn.Sequential(
            nn.Conv1d(ngf, ngf * 2, kernel_size=kernel_size, stride=2, padding=2),
            nn.BatchNorm1d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # size: 70
        self.en3 = nn.Sequential(
            nn.Conv1d(ngf * 2, ngf * 4, kernel_size=kernel_size, stride=2, padding=3),
            nn.BatchNorm1d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # size: 36
        self.en4 = nn.Sequential(
            nn.Conv1d(ngf * 4, ngf * 8, kernel_size=kernel_size, stride=2, padding=2),
            nn.BatchNorm1d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # size: 18
        self.en5 = nn.Sequential(
            nn.Conv1d(ngf * 8, ngf * 8, kernel_size=kernel_size, stride=2, padding=2),
            nn.BatchNorm1d(ngf * 8),
            # nn.Dropout(drop_rate),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # size:  9
        self.en6 = nn.Sequential(
            nn.Conv1d(ngf * 8, 1, kernel_size=kernel_size, stride=2, padding=2),
            nn.Sigmoid()
            # size:  5
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], axis=1)
        x = x.unsqueeze(1)
        en1_out = self.en1(x)
        en2_out = self.en2(en1_out)
        en3_out = self.en3(en2_out)
        en4_out = self.en4(en3_out)
        en5_out = self.en5(en4_out)
        out = self.en6(en5_out)

        return out.squeeze(1)


if __name__ == '__main__':
    x1 = torch.randn(256, 70)
    x2 = torch.randn(256, 70)
    netG = Generator(data_len=70, kernel_size=5)
    netD = Discriminator(data_len=70, kernel_size=5)
    y = netD(x1, x2)


