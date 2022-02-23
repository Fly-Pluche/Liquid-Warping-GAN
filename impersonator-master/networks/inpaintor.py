import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# def get_pad(in_, ksize, stride, atrous=1):
#     out_ = np.ceil(float(in_) / stride)
#     return int(((out_ - 1) * stride + atrous * (ksize - 1) + 1 - in_) / 2)
#
#
# class GatedConv2dWithActivation(nn.Module):
#     """
#     Gated Convlution layer with activation (default activation:LeakyReLU)
#     Params: same as conv2d
#     Input: The feature from last layer "I"
#     Output:\phi(f(I))*\sigmoid(g(I))
#     """
#
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
#                  batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
#         super(GatedConv2dWithActivation, self).__init__()
#         self.batch_norm = batch_norm
#         self.activation = activation
#         self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
#         self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
#                                            bias)
#         self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
#         self.sigmoid = torch.nn.Sigmoid()
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight)
#
#     def gated(self, mask):
#         # return torch.clamp(mask, -1, 1)
#         return self.sigmoid(mask)
#
#     def forward(self, input):
#         x = self.conv2d(input)
#         mask = self.mask_conv2d(input)
#         if self.activation is not None:
#             x = self.activation(x) * self.gated(mask)
#         else:
#             x = x * self.gated(mask)
#         if self.batch_norm:
#             return self.batch_norm2d(x)
#         else:
#             return x
#
#
# class GatedDeConv2dWithActivation(nn.Module):
#     """
#     Gated DeConvlution layer with activation (default activation:LeakyReLU)
#     resize + conv
#     Params: same as conv2d
#     Input: The feature from last layer "I"
#     Output:\phi(f(I))*\sigmoid(g(I))
#     """
#
#     def __init__(self, scale_factor, in_channels, out_channels, kernel_size, stride=1, padding=0,
#                  dilation=1, groups=1, bias=True, batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
#         super(GatedDeConv2dWithActivation, self).__init__()
#         self.conv2d = GatedConv2dWithActivation(in_channels, out_channels, kernel_size, stride, padding,
#                                                 dilation, groups, bias, batch_norm, activation)
#         self.scale_factor = scale_factor
#
#     def forward(self, input):
#         # print(input.size())
#         x = F.interpolate(input, scale_factor=2)
#         return self.conv2d(x)
#
#
# class SelfAttention(nn.Module):
#     """ Self attention Layer"""
#
#     def __init__(self, in_dim, activation, with_attn=False):
#         super(SelfAttention, self).__init__()
#         self.chanel_in = in_dim
#         self.activation = activation
#         self.with_attn = with_attn
#         self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x):
#         """
#             inputs :
#                 x : input feature maps( B X C X W X H)
#             returns :
#                 out : self attention value + input feature
#                 attention: B X N X N (N is Width*Height)
#         """
#         m_batchsize, C, width, height = x.size()
#         proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
#         proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
#         energy = torch.bmm(proj_query, proj_key)  # transpose check
#         attention = self.softmax(energy)  # BX (N) X (N)
#         proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N
#
#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(m_batchsize, C, width, height)
#
#         out = self.gamma * out + x
#         if self.with_attn:
#             return out, attention
#         else:
#             return out


# class InpaintSANet(torch.nn.Module):
#     """
#     Inpaint generator, input should be 5*256*256, where 3*256*256 is the masked image, 1*256*256 for mask, 1*256*256 is the guidence
#     """
#
#     def __init__(self, c_dim=5):
#         super(InpaintSANet, self).__init__()
#         self.c = c_dim
#         cnum = 32
#         self.coarse_net = nn.Sequential(
#             # input is 5*256*256, but it is full convolution network, so it can be larger than 256
#             GatedConv2dWithActivation(c_dim, cnum, 5, 1, padding=get_pad(256, 5, 1)),
#             # downsample 128
#             GatedConv2dWithActivation(cnum, 2 * cnum, 4, 2, padding=get_pad(256, 4, 2)),
#             GatedConv2dWithActivation(2 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),
#             # downsample to 64
#             GatedConv2dWithActivation(2 * cnum, 4 * cnum, 4, 2, padding=get_pad(128, 4, 2)),
#             GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
#             GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
#             # atrous convlution
#             GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
#             GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
#             GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),
#             GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16)),
#             GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
#             # Self_Attn(4*cnum, 'relu'),
#             GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
#             # upsample
#             GatedDeConv2dWithActivation(2, 4 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),
#             # Self_Attn(2*cnum, 'relu'),
#             GatedConv2dWithActivation(2 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),
#             GatedDeConv2dWithActivation(2, 2 * cnum, cnum, 3, 1, padding=get_pad(256, 3, 1)),
#
#             GatedConv2dWithActivation(cnum, cnum // 2, 3, 1, padding=get_pad(256, 3, 1)),
#             # Self_Attn(cnum//2, 'relu'),
#             GatedConv2dWithActivation(cnum // 2, 3, 3, 1, padding=get_pad(128, 3, 1), activation=None)
#         )
#
#         self.refine_conv_net = nn.Sequential(
#             # input is 5*256*256
#             GatedConv2dWithActivation(c_dim, cnum, 5, 1, padding=get_pad(256, 5, 1)),
#             # downsample
#             GatedConv2dWithActivation(cnum, cnum, 4, 2, padding=get_pad(256, 4, 2)),
#             GatedConv2dWithActivation(cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),
#             # downsample
#             GatedConv2dWithActivation(2 * cnum, 2 * cnum, 4, 2, padding=get_pad(128, 4, 2)),
#             GatedConv2dWithActivation(2 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
#             GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
#             GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
#             GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
#             GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
#             # Self_Attn(4*cnum, 'relu'),
#             GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),
#
#             GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16))
#         )
#         self.refine_attn = SelfAttention(4 * cnum, 'relu', with_attn=False)
#         self.refine_upsample_net = nn.Sequential(
#             GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
#
#             GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
#             GatedDeConv2dWithActivation(2, 4 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),
#             GatedConv2dWithActivation(2 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),
#             GatedDeConv2dWithActivation(2, 2 * cnum, cnum, 3, 1, padding=get_pad(256, 3, 1)),
#
#             GatedConv2dWithActivation(cnum, cnum // 2, 3, 1, padding=get_pad(256, 3, 1)),
#             # Self_Attn(cnum, 'relu'),
#             GatedConv2dWithActivation(cnum // 2, 3, 3, 1, padding=get_pad(256, 3, 1), activation=None),
#         )
#
#     def forward(self, imgs, masks, only_out=False, only_x=False):
#         # Coarse
#         masked_imgs = imgs * (1 - masks) + masks
#         if self.c == 5:
#             input_imgs = torch.cat([masked_imgs, masks, torch.full_like(masks, 1.)], dim=1)
#         elif self.c == 4:
#             input_imgs = torch.cat([masked_imgs, masks], dim=1)
#
#         x = self.coarse_net(input_imgs)
#         # if x.size()[3]!=input_imgs.size()[3]:
#         #     x=
#         x = torch.clamp(x, -1., 1.)
#         coarse_x = x
#         # Refine
#         masked_imgs = imgs * (1 - masks) + coarse_x * masks
#         input_imgs = torch.cat([masked_imgs, masks], dim=1)
#         x = self.refine_conv_net(input_imgs)
#         x = self.refine_attn(x)
#         # print(x.size(), attention.size())
#         x = self.refine_upsample_net(x)
#         x = torch.clamp(x, -1., 1.)
#
#         comp_imgs = x * masks + imgs * (1 - masks)
#
#         if only_out:
#             return comp_imgs
#         elif only_x:
#             return x
#         else:
#             return coarse_x, x, comp_imgs



class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # 如果下面这个原论文代码用不了的话，可以换成另一个试试
        out = a_w * a_h*identity
        # out = a_h.expand_as(x) * a_w.expand_as(x) * identity

        return out

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation, with_attn=False):
        super(Self_Attn, self).__init__()
        self.CA=CoordAtt(in_dim,in_dim//8)
        self.chanel_in = in_dim
        self.activation = activation
        self.with_attn = with_attn
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)

        bise=self.CA(x)
        bise=bise.view(m_batchsize, -1, width * height)
        proj_key=torch.add(proj_key,bise)

        energy = torch.bmm(proj_query, proj_key)  # transpose check
        energy = energy / (m_batchsize ** 0.5)
        attention = self.softmax(energy)

        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x

        # out = self.gamma * out + x

        if self.with_attn:
            return out, attention
        else:
            return out

def init_weights(net, init_type='normal', gain=0.02):
    from torch.nn import init
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal(m.weight.data, 1.0, gain)
            init.constant(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def get_pad(in_, ksize, stride, atrous=1):
    out_ = np.ceil(float(in_) / stride)
    return int(((out_ - 1) * stride + atrous * (ksize - 1) + 1 - in_) / 2)


class GCT(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5, mode='l1', after_relu=False):
        super(GCT, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):

        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2, 3), keepdim=True) +
                         self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / \
                   (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)

        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2, 3), keepdim=True) * self.alpha
            norm = self.gamma / \
                   (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)

        gate = 1. + torch.tanh(embedding * norm + self.beta)
        return x*gate


class GatedConv2dWithActivation(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedConv2dWithActivation, self).__init__()
        self.gct = GCT(in_channels)

        self.batch_norm = batch_norm
        self.activation = activation
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                           bias)
        self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def gated(self, mask):
        # return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv2d(input)
        gct_mask = self.gct(input)
        mask = self.mask_conv2d(gct_mask)
        # mask = self.mask_conv2d(input)
        if self.activation is not None:
            # x = self.activation(x) * self.gct(mask)
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x


class GatedDeConv2dWithActivation(torch.nn.Module):
    """
    Gated DeConvlution layer with activation (default activation:LeakyReLU)
    resize + conv
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, scale_factor, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedDeConv2dWithActivation, self).__init__()
        self.conv2d = GatedConv2dWithActivation(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                                groups, bias, batch_norm, activation)
        self.scale_factor = scale_factor

    def forward(self, input):
        # print(input.size())
        x = F.interpolate(input, scale_factor=2)
        return self.conv2d(x)

class InpaintSANet(torch.nn.Module):
    """
    Inpaint generator, input should be 5*256*256, where 3*256*256 is the masked image, 1*256*256 for mask, 1*256*256 is the guidence
    """

    def __init__(self, n_in_channel=5):
        super(InpaintSANet, self).__init__()
        self.n_in_channel = n_in_channel
        cnum = 32
        self.coarse_net = nn.Sequential(
            # input is 5*256*256, but it is full convolution network, so it can be larger than 256
            GatedConv2dWithActivation(n_in_channel, cnum, 5, 1, padding=get_pad(256, 5, 1)),
            # downsample 128
            GatedConv2dWithActivation(cnum, 2 * cnum, 4, 2, padding=get_pad(256, 4, 2)),
            GatedConv2dWithActivation(2 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),
            # downsample to 64
            GatedConv2dWithActivation(2 * cnum, 4 * cnum, 4, 2, padding=get_pad(128, 4, 2)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
            # atrous convlution
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
            # Self_Attn(4*cnum, 'relu'),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
            # upsample
            GatedDeConv2dWithActivation(2, 4 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),
            # Self_Attn(2*cnum, 'relu'),
            GatedConv2dWithActivation(2 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),
            GatedDeConv2dWithActivation(2, 2 * cnum, cnum, 3, 1, padding=get_pad(256, 3, 1)),

            GatedConv2dWithActivation(cnum, cnum // 2, 3, 1, padding=get_pad(256, 3, 1)),
            # Self_Attn(cnum//2, 'relu'),
            GatedConv2dWithActivation(cnum // 2, 3, 3, 1, padding=get_pad(128, 3, 1), activation=None)
        )

        self.refine_conv_net = nn.Sequential(
            # input is 5*256*256
            GatedConv2dWithActivation(n_in_channel, cnum, 5, 1, padding=get_pad(256, 5, 1)),
            # downsample
            GatedConv2dWithActivation(cnum, cnum, 4, 2, padding=get_pad(256, 4, 2)),
            GatedConv2dWithActivation(cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),
            # downsample
            GatedConv2dWithActivation(2 * cnum, 2 * cnum, 4, 2, padding=get_pad(128, 4, 2)),
            GatedConv2dWithActivation(2 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
            # Self_Attn(4*cnum, 'relu'),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),

            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16))
        )
        self.refine_attn = Self_Attn(4 * cnum, 'relu', with_attn=False)
        self.refine_upsample_net = nn.Sequential(
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),

            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedDeConv2dWithActivation(2, 4 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),
            GatedConv2dWithActivation(2 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),
            GatedDeConv2dWithActivation(2, 2 * cnum, cnum, 3, 1, padding=get_pad(256, 3, 1)),

            GatedConv2dWithActivation(cnum, cnum // 2, 3, 1, padding=get_pad(256, 3, 1)),
            # Self_Attn(cnum, 'relu'),
            GatedConv2dWithActivation(cnum // 2, 3, 3, 1, padding=get_pad(256, 3, 1), activation=None),
        )

    def forward(self, imgs, masks, img_exs=None):
        # Coarse
        masked_imgs = imgs * (1 - masks) + masks
        if img_exs == None:
            if self.n_in_channel == 4:
                input_imgs = torch.cat([masked_imgs, masks], dim=1)
            else:
                input_imgs = torch.cat([masked_imgs, masks, torch.full_like(masks, 1.)], dim=1)
        else:
            input_imgs = torch.cat([masked_imgs, img_exs, masks, torch.full_like(masks, 1.)], dim=1)
        # print(input_imgs.size(), imgs.size(), masks.size())
        x = self.coarse_net(input_imgs)
        x = torch.clamp(x, -1., 1.)
        coarse_x = x
        # Refine
        masked_imgs = imgs * (1 - masks) + coarse_x * masks
        if img_exs is None:
            if self.n_in_channel == 4:
                input_imgs = torch.cat([masked_imgs, masks], dim=1)
            else:
                input_imgs = torch.cat([masked_imgs, masks, torch.full_like(masks, 1.)], dim=1)
        else:
            input_imgs = torch.cat([masked_imgs, img_exs, masks, torch.full_like(masks, 1.)], dim=1)
        x = self.refine_conv_net(input_imgs)
        x = self.refine_attn(x)
        # print(x.size(), attention.size())
        x = self.refine_upsample_net(x)
        x = torch.clamp(x, -1., 1.)
        return coarse_x, x