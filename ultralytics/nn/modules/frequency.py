import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_dct import dct_2d,idct_2d
from ultralytics.nn.modules.conv import DWConv,Conv
import torchvision

class DCA_FEM(nn.Module):
    def __init__(self, c, ratio, k):
        super().__init__()
        self.k = k

        # 确保k是平方数（例如k=64对应8x8低频区域）
        self.sqrt_k = int(k ** 0.5)
        assert self.sqrt_k ** 2 == k, "k必须是平方数（如16, 64）"
        
        # 动态下采样比例预测器
        self.down_ratio_predict = nn.Sequential(
            nn.Conv2d(2 * c, c // ratio, 3, stride=2, padding=1),  # 80x80
            nn.SiLU(inplace=True),
            nn.Conv2d(c // ratio, c // ratio, 3, stride=2, padding=1), # 40x40
            nn.SiLU(inplace=True),
            nn.Conv2d(c // ratio, 1, 3, stride=2, padding=1), # 20x20
            nn.AdaptiveAvgPool2d((1,1)), # 1x1
            nn.Sigmoid()  # [B,1,1,1]
        )


        # 多尺度DCT融合模块
        self.multiscale_dct = nn.ModuleDict({
            'avg': DWConv(c, c),
            'mid': DWConv(c, c),
            'low': DWConv(c, c)
        })

        # 通道注意力权重
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * c, c // ratio, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(c // ratio, 2 * c, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1,x2 = x
        b, c, h, w = x1.size()

        # 1、rgb和ir concat
        x = torch.cat([x1,x2],dim=1)


        # 2、Conv+Sigmod 动态下采样
        predict_out = self.down_ratio_predict(x)
        ds_ratio = 0.25 + 0.75 * torch.mean(predict_out)
        # 处理NaN/Inf并限制范围
        if torch.isnan(ds_ratio) or torch.isinf(ds_ratio):
            ds_ratio = torch.tensor(0.25, device=ds_ratio.device)
        ds_ratio = torch.clamp(ds_ratio, 0.25, 1.0)  # 确保在0.25到1.0之间
        # 计算下采样尺寸，确保至少为1
        h_size = max(1, int(torch.round(h * ds_ratio).item()))
        w_size = max(1, int(torch.round(w * ds_ratio).item()))


        # 3、获取下采样系数，对rgb和ir特征进行采样
        x1_down = F.interpolate(x1, size=(h_size, w_size),mode='bilinear')
        x2_down = F.interpolate(x2, size=(h_size, w_size),mode='bilinear')
        

        # 4、对RGB和IR进行DCT变换 + 频率特征多尺度融合变换
        f1 = self.multiscale_dct_fusion(x1_down)
        f2 = self.multiscale_dct_fusion(x2_down)
        feat = torch.cat([f1,f2],dim=1)


        # 5、Conv+Sigmod+残差
        weight = self.gate(feat) # [b,2c,1,1]
        x1_out = x1 * (1 + weight[:,:c,:,:])
        x2_out = x2 * (1 + weight[:,c:,:,:])
        return x1_out, x2_out
    
    def multiscale_dct_fusion(self, x):
        """多尺度DCT特征融合"""
        # 基础DCT计算
        dct_full = dct_2d(x, norm="ortho")
        
        # 多尺度特征提取
        avg = F.adaptive_avg_pool2d(dct_full, (self.sqrt_k, self.sqrt_k))
        mid = dct_full[:, :, ::2, ::2][:, :, :self.sqrt_k, :self.sqrt_k]
        low = dct_full[:, :, :self.sqrt_k, :self.sqrt_k]
        
        # 特征融合
        fused = self.multiscale_dct['avg'](avg) + \
                self.multiscale_dct['mid'](mid) + \
                self.multiscale_dct['low'](low)
        
        return fused




class FFCM(nn.Module):
    def __init__(self, c, ratio, d):
        super().__init__()
        self.d = d

        # 卷积,特征提取
        self.branch1 = nn.Sequential(
            Conv(c, c, k=3),
            Conv(c, c, k=1)
        )
        self.branch2 = nn.Sequential(
            Conv(c, c, k=3),
            Conv(c, c, k=1)
        )

        # 权重
        self.weights = nn.Sequential(
            DWConv(c * 2 , c * 2 // ratio),
            DWConv(c * 2 // ratio, c * 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )

        # 频率增强
        self.dct_conv = Conv(c * 2 , c * 2, k=3, s=1)

    def forward(self, x):
        x1,x2 = x
        b, c, h, w = x1.size()   

        # 1、特征提取
        rgb = self.branch1(x1)
        ir = self.branch2(x2)
        fusion = torch.cat([rgb,ir],dim=1)
        
        # 2、concat，获取权重
        weights = self.weights(fusion)
        fusion_weights = fusion * weights

        # 3、频率增强
        dct_fusion = F.interpolate(fusion_weights, size=(self.d,  self.d), mode='bilinear', align_corners=False)
        dct_feat = dct_2d(dct_fusion, norm="ortho")
        dct_feat_conv = self.dct_conv(dct_feat)
        dct_feat_idct = idct_2d(dct_feat_conv)
        dct_fusion = F.interpolate(dct_feat_idct, size=(h,w), mode='bilinear', align_corners=False)
        out = dct_fusion + fusion

        # out 
        rgb = out[:,:c,:,:]
        ir = out[:,c:,:,:]
        return rgb,ir

# if __name__ == "__main__":
#     x1,x2 = torch.rand(2,64,128,128),torch.rand(2,64,128,128)
#     model = FFCM(64,ratio=16,d=16)
#     print(model((x1,x2))[0].shape) # torch.Size([2, 64, 128, 128])

















# # /**
# # *
# # * author@wk 2025.07.08
# # *
# # */
# class DCTDeformAlign(nn.Module):
#     def __init__(self, in_channels, deform_groups=4, ratio=16):
#         super().__init__()
#         self.in_channels = in_channels
#         # 多尺度特征提取
#         self.conv_1 = Conv(in_channels , in_channels, k=1, s=1)
#         self.conv_3 = Conv(in_channels // 2 , in_channels // 2, k=3, s=1, p=1)
#         self.conv_5 = Conv(in_channels // 2 , in_channels // 2, k=5, s=1, p=2)
 
#         # DCT频域融合
#         self.freq_fusion = nn.Sequential(
#             Conv(in_channels , in_channels, k=1, s=1),
#         )
        
#         # 可变形卷积对齐
#         self.offset_net = nn.Conv2d(in_channels,2*3*3,kernel_size=3,padding=1)
#         self.deform_conv = torchvision.ops.DeformConv2d(
#             in_channels, in_channels, kernel_size=3, padding=1, groups=deform_groups
#         )
#         # 初始化参数
#         nn.init.constant_(self.offset_net.weight, 0)
#         nn.init.constant_(self.offset_net.bias, 0)

#     def forward(self, x):
#         # 1、特征提取
#         x = self.conv_1(x)
#         x_1 = self.conv_3(x[:, :self.in_channels // 2, :, :])
#         x_2 = self.conv_5(x[:, self.in_channels // 2: , :, :])
#         x = torch.cat([x_1,x_2],dim=1)

#         # 2. DCT频域变换
#         x_freq = dct_2d(x, norm='ortho')
#         x_freq = self.freq_fusion(x_freq)
#         aligned_feat = idct_2d(x_freq, norm='ortho')
        
#         # 3. 可变形卷积空间对齐
#         offset = self.offset_net(aligned_feat)
#         aligned_feat = self.deform_conv(aligned_feat, offset)
        
#         return aligned_feat + x


# if __name__ == "__main__":
#     x1,x2 = torch.rand(2,64,20,20),torch.rand(2,64,20,20)
#     model = DCTDeformAlign(64,4)
#     print(model(x1,x2).shape) # torch.Size([2, 64, 128, 128])





# class ChannelAttention(nn.Module):
#     """高效的通道注意力机制"""
#     def __init__(self, in_channels, ratio=16):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
        
#         self.fc = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         return self.sigmoid(out) * x


# class DCTDeformAlign(nn.Module):
#     def __init__(self, in_channels, deform_groups=4, ratio=16):
#         super().__init__()
#         self.in_channels = in_channels
#         # 多尺度特征提取
#         self.conv_1 = Conv(in_channels , in_channels, k=1, s=1)
#         self.conv_3 = Conv(in_channels // 2 , in_channels // 2, k=3, s=1, p=1)
#         self.conv_5 = Conv(in_channels // 2 , in_channels // 2, k=5, s=1, p=2)
 

#         # DCT频域融合
#         self.freq_fusion = nn.Sequential(
#             Conv(in_channels , in_channels, k=1, s=1),
#             ChannelAttention(in_channels),
#             Conv(in_channels , in_channels, k=1, s=1),
#         )
        
#         # 可变形卷积对齐
#         self.offset_net = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels//2, 2*3*3, kernel_size=3, padding=1)
#         )
#         self.deform_conv = torchvision.ops.DeformConv2d(
#             in_channels, in_channels, kernel_size=3, padding=1, groups=deform_groups
#         )
#         # 初始化参数
#         nn.init.constant_(self.offset_net[-1].weight, 0)
#         nn.init.constant_(self.offset_net[-1].bias, 0)

#     def forward(self, x):
#         # 1、特征提取
#         x = self.conv_1(x)
#         x_1 = self.conv_3(x[:, :self.in_channels // 2, :, :])
#         x_2 = self.conv_5(x[:, self.in_channels // 2: , :, :])
#         x = torch.cat([x_1,x_2],dim=1)

#         # 2. DCT频域变换
#         x_freq = dct_2d(x, norm='ortho')
#         x_freq = self.freq_fusion(x_freq)
#         aligned_feat = idct_2d(x_freq, norm='ortho')
        
#         # 3. 可变形卷积空间对齐
#         offset = self.offset_net(aligned_feat)
#         aligned_feat = self.deform_conv(aligned_feat, offset)
        
#         return aligned_feat + x




# if __name__ == "__main__":
#     x1,x2 = torch.rand(2,64,20,20),torch.rand(2,64,20,20)
#     model = DCTDeformAlign(64,4)
#     print(model(x1,x2).shape) # torch.Size([2, 64, 128, 128])



# /**
# *
# * author@wk 2025.07.10
# *
# */
class DCTDeformAlign(nn.Module):
    def __init__(self, in_channels, deform_groups=4, ratio=16):
        super().__init__()
        self.in_channels = in_channels
        # 多尺度特征提取
        self.conv_1 =  nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1),
            nn.GELU()
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,  kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1, padding=2),
            nn.GELU()
        )
 
        # DCT频域融合
        self.freq_fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )
        
        # 可变形卷积对齐
        self.offset_net = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels, 2*3*3, kernel_size=3, padding=1)
        )
        self.deform_conv = torchvision.ops.DeformConv2d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=deform_groups
        )
        # 初始化参数
        nn.init.constant_(self.offset_net[-1].weight, 0)
        nn.init.constant_(self.offset_net[-1].bias, 0)


    def forward(self, x):
        # 1、特征提取
        x = self.conv_1(x)
        x_1 = self.conv_3(x)
        x_2 = self.conv_5(x)
        x_feat = x_1 + x_2

        # 2. DCT频域增强
        x_freq = dct_2d(x_feat, norm='ortho')
        x_freq = self.freq_fusion(x_freq)
        aligned_feat = idct_2d(x_freq, norm='ortho') + x_feat
        
        # 3. 可变形卷积空间对齐
        offset_feat = torch.cat([x, aligned_feat], dim=1)
        offset = self.offset_net(offset_feat)
        aligned_feat = self.deform_conv(aligned_feat, offset)

        return aligned_feat + x
    
if __name__ == "__main__":
    x= torch.rand(2,128,20,20)
    model = DCTDeformAlign(128)
    print(model(x).shape) # torch.Size([2, 64, 128, 128])