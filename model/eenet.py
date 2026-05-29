import torch
import torch.nn as nn


class FrequencyProcessingModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_r = nn.Conv2d(channels, channels, 1)
        self.conv_i = nn.Conv2d(channels, channels, 1)
        self.conv_amp = nn.Conv2d(channels, channels, 1)
        self.conv_phase = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        fft = torch.fft.fft2(x, dim=(-2, -1))
        real, imag = fft.real, fft.imag

        real = self.conv_r(real)
        imag = self.conv_i(imag)

        amp = torch.sqrt(real**2 + imag**2)
        phase = torch.atan2(imag, real)
        amp = self.conv_amp(amp)
        phase = self.conv_phase(phase)

        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        return torch.fft.ifft2(torch.complex(real, imag), dim=(-2, -1)).real


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


class SpatialProcessingModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.res_blocks = nn.Sequential(
            ResidualBlock(channels),
            ResidualBlock(channels),
            ResidualBlock(channels),
        )

    def forward(self, x):
        return self.res_blocks(x)


class DownSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class UpSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels // 2, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class DualDomainInteractionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, f_feat, s_feat):
        return f_feat + self.conv(s_feat), s_feat + self.conv(f_feat)


class EENet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.initial = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.encoder_fpm1 = FrequencyProcessingModule(base_channels)
        self.encoder_spm1 = SpatialProcessingModule(base_channels)
        self.down1 = DownSample(base_channels)

        self.encoder_fpm2 = FrequencyProcessingModule(base_channels * 2)
        self.encoder_spm2 = SpatialProcessingModule(base_channels * 2)
        self.down2 = DownSample(base_channels * 2)

        self.encoder_fpm3 = FrequencyProcessingModule(base_channels * 4)
        self.encoder_spm3 = SpatialProcessingModule(base_channels * 4)

        self.up1 = UpSample(base_channels * 4)
        self.decoder_fpm2 = FrequencyProcessingModule(base_channels * 2)
        self.decoder_spm2 = SpatialProcessingModule(base_channels * 2)

        self.up2 = UpSample(base_channels * 2)
        self.decoder_fpm1 = FrequencyProcessingModule(base_channels)
        self.decoder_spm1 = SpatialProcessingModule(base_channels)

        self.dim1 = DualDomainInteractionModule(base_channels)
        self.dim2 = DualDomainInteractionModule(base_channels * 2)
        self.dim3 = DualDomainInteractionModule(base_channels * 4)

        self.output = nn.Conv2d(base_channels, in_channels, 3, padding=1)

    def forward(self, x):
        x1 = self.initial(x)

        f1 = self.encoder_fpm1(x1)
        s1 = self.encoder_spm1(x1)
        f1, s1 = self.dim1(f1, s1)
        d1 = self.down1(f1 + s1)

        f2 = self.encoder_fpm2(d1)
        s2 = self.encoder_spm2(d1)
        f2, s2 = self.dim2(f2, s2)
        d2 = self.down2(f2 + s2)

        f3 = self.encoder_fpm3(d2)
        s3 = self.encoder_spm3(d2)
        f3, s3 = self.dim3(f3, s3)

        u1 = self.up1(f3 + s3)
        f2d = self.decoder_fpm2(u1 + f2)
        s2d = self.decoder_spm2(u1 + s2)
        f2d, s2d = self.dim2(f2d, s2d)

        u2 = self.up2(f2d + s2d)
        f1d = self.decoder_fpm1(u2 + f1)
        s1d = self.decoder_spm1(u2 + s1)
        f1d, s1d = self.dim1(f1d, s1d)

        return torch.sigmoid(self.output(f1d + s1d))
