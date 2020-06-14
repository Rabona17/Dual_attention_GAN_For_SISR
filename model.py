#This file contains what we proposed as dual_attention Gan with ODE-inspired design.
import math
import torch
from torch import nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.block=[]
        self.block.extend([nn.Conv2d(3, 64, kernel_size=3, padding=1),nn.LeakyReLU(0.2)])
        for i in [64, 128, 256]:
            self.block.extend([nn.Conv2d(i, i, kernel_size=3, stride=2, padding=1),nn.BatchNorm2d(i),nn.LeakyReLU(0.2),nn.Conv2d(i, 2*i, kernel_size=3, padding=1),nn.BatchNorm2d(2*i),nn.LeakyReLU(0.2)])
            
        self.block.extend(
            [nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)]
        )
        self.net = nn.Sequential(
            *self.block
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))

class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)

    
    
class NonLocalBlock2D(nn.Module):
    def __init__(self, in_channels):
        super(NonLocalBlock2D, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = self.in_channels // 2
        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0)
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        #Gaussian
        theta_x = x.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = x.view(batch_size, self.in_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        
        
        z = W_y + x

        return z
    

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, dual_attention):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, in_channels, kernel_size=1)
        self.one_half = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
#         self.one_third = nn.Parameter(torch.FloatTensor([1/3]), requires_grad=True)
#         self.one_sixth = nn.Parameter(torch.FloatTensor([1/6]), requires_grad=True)
        
        if dual_attention:
            self.se=SELayer(in_channels, 8)
            self.spatial_att=NonLocalBlock2D(in_channels)
    def forward(self, x):
        #print(x.shape)
        yn = x
        if hasattr(self, 'spatial_att'):
            #ODE inspired design, here we are using RK4
            G_yn= self.layers(x)
            G_yn= self.se(self.spatial_att(self.lff(G_yn)))

            yn_1 = yn + G_yn
            G_yn_1 = self.layers(yn_1)
            G_yn_1=  self.se(self.spatial_att(self.lff(G_yn_1)))
            return yn+self.one_half*(G_yn+G_yn_1)
        else:
            return yn+self.lff(self.layers(yn))
        

        

########If using 4th order RK, do this part and add attention.      
#         yn = x
#         k1=self.lff(self.layers(yn))
#         k2=self.lff(self.layers(yn+self.one_half*k1))
#         k3=self.lff(self.layers(yn+self.one_half*k2))
#         k4=self.lff(self.layers(yn+k3))
#         yn_1 = yn+self.one_sixth*k1+self.one_third*k2+self.one_third*k3+self.one_sixth*k4
#         return yn_1

#This class fix the code from https://github.com/yjn870/RDN-pytorch/blob/master/models.py, it doesn't work since the paramters in this link is messed up.
class RDN(nn.Module):
    def __init__(self, scale_factor, num_channels, num_features, growth_rate, num_blocks, num_layers, dual_attention):
        super(RDN, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        # shallow feature extraction
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C,dual_attention)])
        for _ in range(self.D - 1):
         
            self.rdbs.append(RDB(self.G0, self.G, self.C, dual_attention))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G0 * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )

        # up-sampling
        assert 2 <= scale_factor <= 4
        if scale_factor == 2 or scale_factor == 4:
            self.upscale = []
            for _ in range(scale_factor // 2):
                self.upscale.extend([nn.Conv2d(self.G0, self.G0 * (2 ** 2), kernel_size=3, padding=3 // 2),
                                     nn.PixelShuffle(2)])
            self.upscale = nn.Sequential(*self.upscale)
        else:
            self.upscale = nn.Sequential(
                nn.Conv2d(self.G0, self.G0 * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
                nn.PixelShuffle(scale_factor)
            )

        self.output = nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2)

    def forward(self, x):
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)
        
        x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
        x = self.upscale(x)
        x = self.output(x)
        return x
    
    
class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x
