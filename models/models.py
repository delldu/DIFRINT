import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pwcNet import PwcNet
import math
import pdb


class UNet2(nn.Module):
    def __init__(self):
        super(UNet2, self).__init__()

        class Encoder(nn.Module):
            def __init__(self, in_nc, out_nc, stride, k_size=3, pad=1):
                super(Encoder, self).__init__()

                self.seq = nn.Sequential(
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(in_nc, out_nc, kernel_size=k_size, stride=stride, padding=0),
                    nn.ReLU(),
                )
                self.GateConv = nn.Sequential(
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(in_nc, out_nc, kernel_size=k_size, stride=stride, padding=0),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                return self.seq(x) * self.GateConv(x)

        class Decoder(nn.Module):
            def __init__(self, in_nc, out_nc, stride, k_size=3, pad=1, tanh=False):
                super(Decoder, self).__init__()

                self.seq = nn.Sequential(
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(in_nc, in_nc, kernel_size=k_size, stride=stride, padding=0),
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(in_nc, out_nc, kernel_size=k_size, stride=stride, padding=0),
                )

                if tanh:
                    self.activ = nn.Tanh()
                else:
                    self.activ = nn.ReLU()

                self.GateConv = nn.Sequential(
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(in_nc, in_nc, kernel_size=k_size, stride=stride, padding=0),
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(in_nc, out_nc, kernel_size=k_size, stride=stride, padding=0),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                s = self.seq(x)
                s = self.activ(s)
                return s * self.GateConv(x)

        self.enc0 = Encoder(16, 32, stride=1)
        self.enc1 = Encoder(32, 32, stride=2)
        self.enc2 = Encoder(32, 32, stride=2)
        self.enc3 = Encoder(32, 32, stride=2)

        self.dec0 = Decoder(32, 32, stride=1)
        # up-scaling + concat
        self.dec1 = Decoder(32 + 32, 32, stride=1)
        self.dec2 = Decoder(32 + 32, 32, stride=1)
        self.dec3 = Decoder(32 + 32, 32, stride=1)

        self.dec4 = Decoder(32, 3, stride=1, tanh=True)

    def forward(self, w1, w2, flow1, flow2, fr1, fr2):
        # fr1, fr2 -- frame_prev, frame_next
        s0 = self.enc0(torch.cat([w1, w2, flow1, flow1, fr1, fr2], dim=1).cuda())
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)

        s4 = self.dec0(s3)
        # up-scaling + concat
        s4 = F.interpolate(s4, scale_factor=2, mode="nearest")
        s5 = self.dec1(torch.cat([s4, s2], 1).cuda())
        s5 = F.interpolate(s5, scale_factor=2, mode="nearest")
        s6 = self.dec2(torch.cat([s5, s1], 1).cuda())
        s6 = F.interpolate(s6, scale_factor=2, mode="nearest")
        s7 = self.dec3(torch.cat([s6, s0], 1).cuda())

        out = self.dec4(s7)
        return out


class DIFNet2(nn.Module):
    def __init__(self):
        super(DIFNet2, self).__init__()

        class Backward(nn.Module):
            def __init__(self):
                super(Backward, self).__init__()

            def forward(self, tensorInput, tensorFlow, scale=1.0):
                # xxxx8888
                # hasattr(self, 'tensorPartial') -- False
                if (
                    hasattr(self, "tensorPartial") == False
                    or self.tensorPartial.size(0) != tensorFlow.size(0)
                    or self.tensorPartial.size(2) != tensorFlow.size(2)
                    or self.tensorPartial.size(3) != tensorFlow.size(3)
                ):
                    self.tensorPartial = (
                        torch.FloatTensor()
                        .resize_(tensorFlow.size(0), 1, tensorFlow.size(2), tensorFlow.size(3))
                        .fill_(1.0)
                        .cuda()
                    )
                # hasattr(self, 'tensorGrid') -- False
                if (
                    hasattr(self, "tensorGrid") == False
                    or self.tensorGrid.size(0) != tensorFlow.size(0)
                    or self.tensorGrid.size(2) != tensorFlow.size(2)
                    or self.tensorGrid.size(3) != tensorFlow.size(3)
                ):
                    tensorHorizontal = (
                        torch.linspace(-1.0, 1.0, tensorFlow.size(3))
                        .view(1, 1, 1, tensorFlow.size(3))
                        .expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
                    )
                    tensorVertical = (
                        torch.linspace(-1.0, 1.0, tensorFlow.size(2))
                        .view(1, 1, tensorFlow.size(2), 1)
                        .expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))
                    )

                    self.tensorGrid = torch.cat([tensorHorizontal, tensorVertical], dim=1).cuda()
                # pdb.set_trace()
                tensorInput = torch.cat([tensorInput, self.tensorPartial], dim=1)
                tensorFlow = torch.cat(
                    [
                        tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
                        tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0),
                    ],
                    1,
                )

                tensorOutput = F.grid_sample(
                    input=tensorInput,
                    grid=(self.tensorGrid + tensorFlow * scale).permute(0, 2, 3, 1),
                    mode="bilinear",
                    padding_mode="zeros",
                )

                # tensorOutput.size() -- [1, 4, 720, 1280]
                tensorMask = tensorOutput[:, -1:, :, :]
                # tensorMask.size() -- [1, 1, 720, 1280]
                tensorMask[tensorMask > 0.999] = 1.0
                tensorMask[tensorMask <= 0.999] = 0.0

                return tensorOutput[:, :-1, :, :] * tensorMask

        # PWC
        self.pwc = PwcNet()
        # self.pwc.load_state_dict(torch.load('./trained_models/sintel.pytorch'))
        self.pwc.eval()

        # Warping layer
        self.warpLayer = Backward()
        self.warpLayer.eval()

        # UNets
        self.UNet2 = UNet2()
        self.ResNet2 = ResNet2()

    def warpFrame(self, fr_1, fr_2, scale=1.0):
        with torch.no_grad():
            temp_w = int(math.floor(math.ceil(fr_1.size(3) / 64.0) * 64.0))  # Due to Pyramid method?
            temp_h = int(math.floor(math.ceil(fr_1.size(2) / 64.0) * 64.0))

            temp_fr_1 = F.interpolate(input=fr_1, size=(temp_h, temp_w), mode="nearest")
            temp_fr_2 = F.interpolate(input=fr_2, size=(temp_h, temp_w), mode="nearest")

            flow = 20.0 * F.interpolate(
                input=self.pwc(temp_fr_1, temp_fr_2),
                size=(fr_1.size(2), fr_1.size(3)),
                mode="bilinear",
                align_corners=False,
            )
            return self.warpLayer(fr_2, flow, scale), flow

    def forward(self, frame_prev, frame_curr, frame_next):
        w1, flow1 = self.warpFrame(frame_next, frame_prev, scale=0.5)
        w2, flow2 = self.warpFrame(frame_prev, frame_next, scale=0.5)

        I_int = self.UNet2(w1, w2, flow1, flow2, frame_prev, frame_next)
        f_int, flo_int = self.warpFrame(I_int, frame_curr)

        fhat = self.ResNet2(I_int, f_int, flo_int, frame_curr)
        # return fhat, I_int
        return fhat

class ResNet2(nn.Module):
    def __init__(self):
        super(ResNet2, self).__init__()

        class ConvBlock(nn.Module):
            def __init__(self, in_ch, out_ch):
                super(ConvBlock, self).__init__()

                self.seq = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0), nn.ReLU())

                self.GateConv = nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=0),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=0),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                return self.seq(x) * self.GateConv(x)

        class ResBlock(nn.Module):
            def __init__(self, num_ch):
                super(ResBlock, self).__init__()

                self.seq = nn.Sequential(nn.Conv2d(num_ch, num_ch, kernel_size=1, stride=1, padding=0), nn.ReLU())

                self.GateConv = nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(num_ch, num_ch, kernel_size=3, stride=1, padding=0),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(num_ch, num_ch, kernel_size=3, stride=1, padding=0),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                return self.seq(x) * self.GateConv(x) + x

        self.seq = nn.Sequential(
            ConvBlock(11, 32),
            ResBlock(32),
            ResBlock(32),
            ResBlock(32),
            ResBlock(32),
            ResBlock(32),
            ConvBlock(32, 3),
            nn.Tanh(),
        )

    def forward(self, I_int, f_int, flo_int, f3):
        return self.seq(torch.cat([I_int, f_int, flo_int, f3], dim=1).cuda())
