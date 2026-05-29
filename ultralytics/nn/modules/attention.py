import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ultralytics.nn.modules.conv import DWConv,Conv

# 1、位置编码
class PositionEncoding(nn.Module):
    def __init__(self, model_dim,max_len=6400):
        super(PositionEncoding, self).__init__()

        # 1、定义位置编码矩阵
        encoding = torch.zeros(max_len, model_dim)  # 6400 x model_dim

        # 2、位置参数
        position = torch.arange(0, max_len).float().unsqueeze(1)
        index = torch.arange(0,model_dim,2)

        # 3、生成位置编码矩阵
        encoding[:, 0::2] = torch.sin(position / (10000 ** (index/model_dim)))  # 偶数列使用sin函数
        encoding[:, 1::2] = torch.cos(position / (10000 ** (index/model_dim)))  # 奇数列使用cos函数

        # 4、encoding矩阵前添加一个维度batch，并设置梯度不更新
        encoding = encoding.unsqueeze(0)
        self.register_buffer("encoding", encoding)  # 训练过程中不会更新，多gpu时不会报错

    def forward(self, x):
        x = x + self.encoding[:, : x.size(1), :]
        return x



# 2、FFN网络
class FeedForward(nn.Module):
    def __init__(self, in_channel, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(in_channel, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, in_channel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x





# 3、多头注意力机制：Encoder中的注意力机制不需要mask
class MultiHeadAttention(nn.Module):
    """
        input_dim:输入token的维度
        num_heads:注意力头数
    """
    def __init__(self, input_channel, head_nums, attntion_dropout=0.1):
        super().__init__()
        self.head_channel = input_channel // head_nums  # 每一个head的token维度
        self.head_nums = head_nums  # 注意力头数
        
        # 输入
        self.query = nn.Linear(input_channel, input_channel)
        self.key = nn.Linear(input_channel, input_channel)
        self.value = nn.Linear(input_channel, input_channel)

        # 输出
        self.out_proj = nn.Linear(input_channel, input_channel)

        # Dropout
        self.attention_dropout = nn.Dropout(attntion_dropout)


    def forward(self,query,key,value):
        # 1、获取token维度
        batch, seq_len, _ = query.size()  # batch,token数量,token的维度

        # 2、获取QKV
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)

        # 3、转化为多头QKV
        # (batch,seq_len,input_channel) -> (batch,head_num,seq_len,head_channel)
        q_state = Q.view(batch, seq_len, self.head_nums, self.head_channel).transpose(1,2)
        k_state = K.view(batch, seq_len, self.head_nums, self.head_channel).transpose(1,2)
        v_state = V.view(batch, seq_len, self.head_nums, self.head_channel).transpose(1,2)

        # 4、注意力计算
        attentions_weight = q_state @ k_state.transpose(-1, 2) / math.sqrt(self.head_channel)
        attentions_weight = torch.softmax(attentions_weight, dim=-1) 
        attentions_weight = self.attention_dropout(attentions_weight)
        score = attentions_weight @ v_state

        # 5、拼接计算好的注意力分数
        score = score.transpose(1,2).contiguous().view(batch, seq_len, -1)
        score = self.out_proj(score)

        return score




# 4、EncoderLayer
class EncoderLayer(nn.Module):
    def __init__(self, in_channel, head_nums, hidden_dim, dropout):
        super(EncoderLayer, self).__init__()
        # 多头注意力
        self.attention = MultiHeadAttention(in_channel, head_nums,dropout)  
        self.norm1 = nn.LayerNorm(in_channel)  # 注意力后的LayerNorm
        self.drop1 = nn.Dropout(dropout)

        # FFN
        self.ff = FeedForward(in_channel, hidden_dim, dropout)  # ff
        self.norm2 = nn.LayerNorm(in_channel)  # ff之后的LayerNorm
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        # 1、先进行attention计算s
        input = x
        x = self.attention(
            query = x,
            key = x,
            value = x
        )
        x = self.drop1(x)
        x = self.norm1(x + input)

        # 2、经过ff
        input = x
        x = self.ff(x)
        x = self.drop2(x)
        x = self.norm2(x + input)
        return x




# 5、DecoderLayer
class DecoderLayer(nn.Module):
    def __init__(self, in_channel, head_nums, hidden_dim, dropout):
        super(DecoderLayer, self).__init__()
        # 多头注意力
        self.attention = MultiHeadAttention(in_channel, head_nums,dropout)  
        self.norm1 = nn.LayerNorm(in_channel)  # 注意力后的LayerNorm
        self.drop1 = nn.Dropout(dropout)

        # FFN
        self.ff = FeedForward(in_channel, hidden_dim, dropout)  # ff
        self.norm2 = nn.LayerNorm(in_channel)  # ff之后的LayerNorm
        self.drop2 = nn.Dropout(dropout)

    def forward(self, q, k, v):
        # 1、先进行attention计算
        input = q
        x = self.attention(
            query = q,
            key = k,
            value = v
        )
        x = self.drop1(x)
        x = self.norm1(x + input)

        # 2、经过ff
        input = x
        x = self.ff(x)
        x = self.drop2(x)
        x = self.norm2(x + input)
        return x




##################################################################################################
# 一、TransformerEncoder
class TransformerEncoder(nn.Module):
    """
        input_channel:输入token维度
        head_nums:注意力头的数量
        num_layers:TransformerEncoderLayer需要迭代的次数
        hidden_dim:ff中隐藏层的层数
    """
    def __init__(
        self, input_channel, head_nums, num_layers, hidden_dim, dropout=0.1
    ):
        super(TransformerEncoder, self).__init__()
        # 位置编码
        self.position_encoding = PositionEncoding(input_channel)  
        #多层attention计算
        self.layers = nn.ModuleList(
            [
                EncoderLayer(input_channel, head_nums, hidden_dim, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        # 1、加入位置编码特征
        x = self.position_encoding(x)

        # 2、特征丢入Encoder中,进行N次迭代计算，输出token
        for layer in self.layers:
            x = layer(x)
        return x




# 二、TransformerDecoder
class TransformerDecoder(nn.Module):
    def __init__(
        self, input_channel, head_nums, num_layers, hidden_dim, dropout=0.1
    ):
        super(TransformerDecoder, self).__init__()
        # 位置编码
        self.position_encoding = PositionEncoding(input_channel)  
        #多层attention计算
        self.layers = nn.ModuleList(
            [
                DecoderLayer(input_channel, head_nums, hidden_dim, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, q, k, v):
        # 1、加入位置编码特征
        q = self.position_encoding(q)
        k = self.position_encoding(k)
        v = self.position_encoding(v)

        # 2、特征丢入Encoder中,进行N次迭代计算，输出token
        for layer in self.layers:
            q = layer(q,k,v)
        return q

##################################################################################################
class SpatialAttentionBlock(nn.Module):
    def __init__(self, chs, head_nums, num_layers, target_size,ratio=8):
        """
        chs: 三个输入特征图的通道数，格式为(ch_x, ch_y, ch_z)
        target_size: 池化后的目标尺寸（正方形）
        """
        super().__init__()
        self.target_size = target_size
        ch_x, ch_y, ch_z = chs
        self.total_ch = sum(chs)
        
        # 池化
        self.feat_80 = nn.Sequential(
            DWConv(ch_x , ch_x // ratio),
            nn.SiLU(),
            DWConv(ch_x // ratio, ch_x),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )
        self.feat_40 = nn.Sequential(
            DWConv(ch_y , ch_y // ratio),
            nn.SiLU(),
            DWConv(ch_y // ratio, ch_y),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )
        self.feat_20 = nn.Sequential(
            DWConv(ch_z , ch_z // ratio),
            nn.SiLU(),
            DWConv(ch_z // ratio, ch_z),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )


        # DWConv
        self.dwConv = DWConv(self.total_ch,self.total_ch)

        # 分支1: 1x1 + 3x3 + 1x1卷积
        self.branch1 = nn.Sequential(
            DWConv(self.total_ch, self.total_ch // 2, k=1),
            DWConv(self.total_ch // 2, self.total_ch // 2, k=3),
            DWConv(self.total_ch // 2, self.total_ch , k=1),
        )
        
        # 分支2: 1x1卷积 + 自注意力
        self.branch2_conv =  DWConv(self.total_ch, self.total_ch, k=1)
        self.branch2 = nn.Sequential(
            TransformerEncoder(self.total_ch,head_nums,num_layers,hidden_dim=self.total_ch*2)
        )
        
        # 最终融合的1x1卷积（可选）
        self.final_conv =  DWConv(self.total_ch, self.total_ch, k=1)
        
        # 保存原始通道数用于分割
        self.ch_x = ch_x
        self.ch_y = ch_y
        self.ch_z = ch_z

    def forward(self, x, y, z):
        # 保存原始尺寸
        orig_sizes = {
            'x': x.shape[2:],
            'y': y.shape[2:],
            'z': z.shape[2:]
        }
        
        # 1、统一池化到目标尺寸
        def pool_features(feature,weight):
            avg_pool = F.adaptive_avg_pool2d(feature, self.target_size)
            max_pool = F.adaptive_max_pool2d(feature, self.target_size)
            return avg_pool * weight + max_pool * (1 - weight)
        x_weight = self.feat_80(x)
        y_weight = self.feat_40(y)
        z_weight = self.feat_20(z)

        x_pool = pool_features(x,x_weight)
        y_pool = pool_features(y,y_weight)
        z_pool = pool_features(z,z_weight)
        
        # 2、DWConv
        fused = torch.cat([x_pool, y_pool, z_pool], dim=1)
        fused = self.dwConv(fused)
        B,C,W,H = fused.shape


        ###############################################
        # 3、双分支处理
        ###############################################
        # 分支一
        b1 = self.branch1(fused)
        # 分支二
        fused = self.branch2_conv(fused)
        fused_feat = fused.reshape(B, C, -1).permute(0, 2, 1).contiguous()
        b2 = self.branch2(fused_feat)
        b2 = b2.reshape(B, W, H, C).permute(0, 3, 1, 2).contiguous()

        


        # 4、特征融合
        fused_out = self.final_conv(b1 + b2)
        
        # 按原始通道数分割
        x_out, y_out, z_out = torch.split(
            fused_out, 
            [self.ch_x, self.ch_y, self.ch_z], 
            dim=1
        )
        
        # 上采样到原始尺寸并残差连接
        def resize_and_add(orig, output, orig_size):
            output = F.interpolate(output, size=orig_size, 
                                  mode='bilinear', align_corners=False)
            return orig + output
        
        x_final = resize_and_add(x, x_out, orig_sizes['x'])
        y_final = resize_and_add(y, y_out, orig_sizes['y'])
        z_final = resize_and_add(z, z_out, orig_sizes['z'])
        
        return x_final, y_final, z_final
# if __name__ == "__main__":
#     x = torch.rand(2,16,80,80)
#     y = torch.rand(2,32,40,40)
#     z = torch.rand(2,64,20,20)
#     model = SEM((16,32,64))
#     print(model(x,y,z)[2].shape) # torch.Size([2, 64, 40, 40])    
##################################################################################################



class SEMBlock(nn.Module):
    def __init__(self, chs, head_nums, num_layers, target_size,ratio=8):
        super().__init__()
        self.target_size = target_size
        ch_x, ch_y, ch_z = chs
        self.ch_x = ch_x
        self.ch_y = ch_y
        self.ch_z = ch_z
        self.total_ch = sum(chs) // 2

        # 池化
        self.feat_80 = nn.Sequential(
            DWConv(ch_x , ch_x // ratio),
            DWConv(ch_x // ratio, ch_x),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )
        self.feat_40 = nn.Sequential(
            DWConv(ch_y , ch_y // ratio),
            DWConv(ch_y // ratio, ch_y),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )
        self.feat_20 = nn.Sequential(
            DWConv(ch_z , ch_z // ratio),
            DWConv(ch_z // ratio, ch_z),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )



        # DWConv
        self.dwConv = Conv(self.total_ch * 2, self.total_ch, k=1, s=1)

        # 分支一：多尺度卷积
        self.conv_1_1 = Conv(self.total_ch,self.total_ch,k=1,s=1)
        self.conv_multi_1 = Conv(self.total_ch,self.total_ch ,k=1,s=1)
        self.conv_multi_2 = Conv(self.total_ch ,self.total_ch ,k=3,s=1,p=1)
        self.conv_multi_3 = Conv(self.total_ch ,self.total_ch ,k=5,s=1,p=2)
        self.conv_mlp = nn.Sequential(
            DWConv(self.total_ch  , self.total_ch // ratio),
            DWConv(self.total_ch // ratio, self.total_ch ),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )
        self.conv_1_2 = Conv(self.total_ch,self.total_ch,k=1,s=1)


        # 分支二：自注意力
        self.conv_2_1 = Conv(self.total_ch,self.total_ch,k=1,s=1)
        self.attntion = TransformerEncoder(self.total_ch, head_nums,num_layers,hidden_dim=self.total_ch)
        self.conv_2_2 = Conv(self.total_ch, self.total_ch,k=1,s=1)

        # output
        self.conv_output = Conv(self.total_ch, self.total_ch * 2, k=1, s=1)



    def forward(self, x, y, z):
        # 保存原始尺寸
        orig_sizes = {
            'x': x.shape[2:],
            'y': y.shape[2:],
            'z': z.shape[2:]
        }


        # 1、下采样
        def pool_features(feature,weight):
            avg_pool = F.adaptive_avg_pool2d(feature, self.target_size)
            max_pool = F.adaptive_max_pool2d(feature, self.target_size)
            return avg_pool * weight + max_pool * (1 - weight)
        x_weight = self.feat_80(x)
        y_weight = self.feat_40(y)
        z_weight = self.feat_20(z)

        x_pool = pool_features(x,x_weight)
        y_pool = pool_features(y,y_weight)
        z_pool = pool_features(z,z_weight)

        # 2、DWConv,降通道维度 // 2
        out = torch.cat([x_pool, y_pool, z_pool], dim=1)
        out = self.dwConv(out)

        # 3、分支
        # 3.1 分支一：多尺度卷积
        out_1 = self.conv_1_1(out)
        out_1 = self.conv_multi_1(out_1) + self.conv_multi_2(out_1) + self.conv_multi_3(out_1)
        out_1 = self.conv_mlp(out_1) * out_1 # 通道权重
        out_1 = self.conv_1_2(out_1)

        # 3.2 分支二：自注意力
        out_2 = self.conv_2_1(out)
        B,C,W,H = out_2.shape
        feat = out_2.reshape(B, C, -1).permute(0, 2, 1).contiguous()
        feat_attention = self.attntion(feat)
        out_2 = feat_attention.reshape(B, W, H, C).permute(0, 3, 1, 2).contiguous() + out_2
        out_2 = self.conv_2_2(out_2)

        # 4、output，升通道维度 * 2
        output = out_1 + out_2
        output = self.conv_output(output)

        # 5、按原始通道数分割，并上采样到原始维度
        x_out, y_out, z_out = torch.split(
            output, 
            [self.ch_x, self.ch_y, self.ch_z], 
            dim=1
        )
        # 上采样到原始尺寸并残差连接
        def resize_and_add(orig, output, orig_size):
            output = F.interpolate(output, size=orig_size, 
                                  mode='bilinear', align_corners=False)
            return orig + output
        x_final = resize_and_add(x, x_out, orig_sizes['x'])
        y_final = resize_and_add(y, y_out, orig_sizes['y'])
        z_final = resize_and_add(z, z_out, orig_sizes['z'])

        return x_final, y_final, z_final

# if __name__ == "__main__":
#     x = torch.rand(2,16,80,80)
#     y = torch.rand(2,32,40,40)
#     z = torch.rand(2,64,20,20)
#     model = SEMBlock((16,32,64),head_nums=8,num_layers=2,target_size=16)
#     print(model(x,y,z)[0].shape) # torch.Size([2, 64, 40, 40])















##########################################################
# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, ratio=16):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
        
#         self.mlp = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // ratio, 1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels // ratio, in_channels, 1)
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.mlp(self.avg_pool(x))
#         max_out = self.mlp(self.max_pool(x))
#         attention = self.sigmoid(avg_out + max_out)
#         return x * attention
    


# class ChannelCorssAttntion(nn.Module):
#     def __init__(self, input_channels, head_nums, num_layers):
#         super().__init__()
        
#         # conv
#         self.conv_1 = Conv(input_channels,input_channels,k=1,s=1)
#         self.ca = ChannelAttention(input_channels)
#         self.conv_3 = Conv(input_channels,input_channels,k=3,s=1,p=1)

#         # CSA
#         self.csa = TransformerDecoder(
#             input_channel=input_channels,
#             head_nums=head_nums,
#             num_layers=num_layers,
#             hidden_dim=input_channels * 2)
        
#         # out
#         self.Conv = Conv(input_channels , input_channels, k=1)

#     def forward(self,original_feat,pooling_feat,assisted_feat):
#         _, _, W_original, H_original = original_feat.shape
#         B, C, W, H = pooling_feat.shape
#         # 1、CA
#         x_ca = self.conv_1(original_feat)
#         x_ca = self.ca(x_ca)
#         x_ca = self.conv_3(x_ca)

#         # 2、CSA
#         pooling_feat = pooling_feat.reshape(B, C, -1).permute(0, 2, 1).contiguous()
#         assisted_feat = assisted_feat.reshape(B, C, -1).permute(0, 2, 1).contiguous()
#         attention = self.csa(pooling_feat,assisted_feat,assisted_feat)
#         x_csa = attention.reshape(B, W, H, C).permute(0, 3, 1, 2).contiguous()
#         x_csa = F.interpolate(x_csa, size=(W_original,H_original), mode='bilinear', align_corners=False)

#         # 3、output
#         x_out = x_ca + x_csa
#         x_out = self.Conv(x_out)
#         return x_out

##############################################################################################
# # /**
# # *
# # * author@wk 2025.07.10
# # *
# # */
# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, ratio=16):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
        
#         self.mlp = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // ratio, 1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels // ratio, in_channels, 1)
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.mlp(self.avg_pool(x))
#         max_out = self.mlp(self.max_pool(x))
#         attention = self.sigmoid(avg_out + max_out)
#         return x * attention
    


# class ChannelCorssAttntion(nn.Module):
#     def __init__(self, input_channels, head_nums, num_layers):
#         super().__init__()
        
#         # conv
#         self.conv_1 = nn.Sequential(
#             nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1),
#             nn.GELU()
#         )
#         self.ca = ChannelAttention(input_channels)
#         self.conv_3 =  nn.Sequential(
#             nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1),
#             nn.GELU()
#         )

#         # CSA
#         self.csa = TransformerDecoder(
#             input_channel=input_channels,
#             head_nums=head_nums,
#             num_layers=num_layers,
#             hidden_dim=input_channels * 2)
        

#     def forward(self,original_feat,pooling_feat,assisted_feat):
#         _, _, W_original, H_original = original_feat.shape
#         B, C, W, H = pooling_feat.shape
#         # 1、CA
#         x_ca = self.conv_1(original_feat)
#         x_ca = self.ca(x_ca)
#         x_ca = self.conv_3(x_ca)

#         # 2、CSA
#         pooling_feat = pooling_feat.reshape(B, C, -1).permute(0, 2, 1).contiguous()
#         assisted_feat = assisted_feat.reshape(B, C, -1).permute(0, 2, 1).contiguous()
#         attention = self.csa(pooling_feat,assisted_feat,assisted_feat)
#         x_csa = attention.reshape(B, W, H, C).permute(0, 3, 1, 2).contiguous()
#         x_csa = F.interpolate(x_csa, size=(W_original,H_original), mode='bilinear', align_corners=False)

#         # 3、output
#         x_out = x_ca + x_csa
#         return x_out

###############################################################################################
# /**
# *
# * author@wk 2025.07.13
# *
# */
# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, ratio=16):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
        
#         self.mlp = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // ratio, 1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels // ratio, in_channels, 1)
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.mlp(self.avg_pool(x))
#         max_out = self.mlp(self.max_pool(x))
#         attention = self.sigmoid(avg_out + max_out)
#         return x * attention
    


# class ChannelCorssAttntion(nn.Module):
#     def __init__(self, input_channels, head_nums, num_layers):
#         super().__init__()
        
#         # conv
#         self.conv_1 = nn.Sequential(
#             nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1),
#             nn.GELU()
#         )
#         self.ca = ChannelAttention(input_channels)
#         self.conv_3 = nn.Sequential(
#             nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1),
#             nn.GELU()
#         )
        
#         # CSA
#         self.csa = TransformerDecoder(
#             input_channel=input_channels,
#             head_nums=head_nums,
#             num_layers=num_layers,
#             hidden_dim=input_channels * 2)
        

#     def forward(self,original_feat,pooling_feat,assisted_feat):
#         _, _, W_original, H_original = original_feat.shape
#         B, C, W, H = pooling_feat.shape
#         # 1、CA
#         x_ca = self.conv_1(original_feat)
#         x_ca = self.ca(x_ca)
#         x_ca = self.conv_3(x_ca)

#         # 2、CSA
#         pooling_feat = pooling_feat.reshape(B, C, -1).permute(0, 2, 1).contiguous()
#         assisted_feat = assisted_feat.reshape(B, C, -1).permute(0, 2, 1).contiguous()
#         attention = self.csa(pooling_feat,assisted_feat,assisted_feat)
#         x_csa = attention.reshape(B, W, H, C).permute(0, 3, 1, 2).contiguous()
#         x_csa = F.interpolate(x_csa, size=(W_original,H_original), mode='bilinear', align_corners=False)

#         # 3、output
#         x_out = x_ca + x_csa
#         return x_out
###############################################################################################


# class SelfAndCorssAttntion(nn.Module):
#     def __init__(self, input_channels, head_nums, num_layers):
#         super().__init__()

#         # 自注意力
#         self.sa = TransformerEncoder(
#             input_channel=input_channels,
#             head_nums=head_nums,
#             num_layers=num_layers,
#             hidden_dim=input_channels * 2 
#         )
        
#         # 交叉注意力
#         self.ca = TransformerDecoder(
#             input_channel=input_channels,
#             head_nums=head_nums,
#             num_layers=num_layers,
#             hidden_dim=input_channels * 2
#         )

#         # fusion conv
#         self.fusion_conv = nn.Sequential(
#             nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1,padding=1),
#             nn.GELU(),
#             nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1),
#             nn.GELU(),
#             nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1),
#             nn.GELU(),
#             nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1)
#         )
        

#     def forward(self,rgb,ir,original_rgb):
#         B, C, W, H = rgb.shape
#         _, _, original_W, original_H = original_rgb.shape

#         rgb = rgb.reshape(B, C, -1).permute(0, 2, 1).contiguous()
#         ir = ir.reshape(B, C, -1).permute(0, 2, 1).contiguous()
       
#         # 1、自注意力
#         x_sa = self.sa(rgb)
#         x_sa = x_sa.reshape(B, W, H, C).permute(0, 3, 1, 2).contiguous()
#         x_sa = F.interpolate(x_sa, size=(original_W,original_H), mode='bilinear', align_corners=False)

#         # 2、交叉注意力
#         x_ca = self.ca(rgb,ir,ir)
#         x_ca = x_ca.reshape(B, W, H, C).permute(0, 3, 1, 2).contiguous()
#         x_ca = F.interpolate(x_ca, size=(original_W,original_H), mode='bilinear', align_corners=False)

#         # 3、fusion
#         x_out = x_sa + x_ca
#         x_out = self.fusion_conv(x_out)
#         return x_out


########################################################################################

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention
    

class ChannelCorssAttntion(nn.Module):
    def __init__(self, input_channels, head_nums, num_layers):
        super().__init__()
        
        # conv
        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1),
            nn.GELU()
        )
        self.ca = ChannelAttention(input_channels)
        self.conv_3 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1,padding=1),
            nn.GELU()
        )
        
        # CSA
        self.csa = TransformerDecoder(
            input_channel=input_channels,
            head_nums=head_nums,
            num_layers=num_layers,
            hidden_dim=input_channels * 2)

        # out Conv
        # self.conv_out = nn.Sequential(
        #     nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1),
        #     nn.GELU()
        # )
    
        

    def forward(self,original_feat,pooling_feat,assisted_feat):
        _, _, W_original, H_original = original_feat.shape
        B, C, W, H = pooling_feat.shape
        # 1、CA
        x_ca = self.conv_1(original_feat)
        x_ca = self.ca(x_ca)
        x_ca = self.conv_3(x_ca)

        # 2、CSA
        pooling_feat = pooling_feat.reshape(B, C, -1).permute(0, 2, 1).contiguous()
        assisted_feat = assisted_feat.reshape(B, C, -1).permute(0, 2, 1).contiguous()
        attention = self.csa(pooling_feat,assisted_feat,assisted_feat)
        x_csa = attention.reshape(B, W, H, C).permute(0, 3, 1, 2).contiguous()
        x_csa = F.interpolate(x_csa, size=(W_original,H_original), mode='bilinear', align_corners=False)

        # 3、output
        x_out = x_ca + x_csa
        # x_out = self.conv_out(x_out)
        return x_out










        
# if __name__ == "__main__":
#     x = torch.rand(2,16,80,80)
#     y = torch.rand(2,16,20,20)
#     z = torch.rand(2,16,20,20)
#     model = ChannelCorssAttntion(input_channels=16,head_nums=2,num_layers=2)
#     print(model(x,y,z).shape) # torch.Size([2, 64, 40, 40])