import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

# class DNN(nn.Module):
#     def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
#         super(DNN, self).__init__()
#         self.in_dims = in_dims
#         self.out_dims = out_dims
#         assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
#         self.time_type = time_type
#         self.time_emb_dim = emb_size
#         self.norm = norm

#         self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

#         if self.time_type == "cat":
#             in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
#         else:
#             raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
#         out_dims_temp = self.out_dims
        
#         self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
#             for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
#         self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
#             for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        
#         # emb_out_linear: 输入维度应该是 out_layers 的输出维度 (out_dims[-1])
#         # 输出维度是 in_dims[0] (原始输入维度)
#         self.emb_out_linear = nn.Linear(self.out_dims[-1], self.in_dims[0])

#         self.drop = nn.Dropout(dropout)
#         self.init_weights()
    
#     def init_weights(self):
#         for layer in self.in_layers:
#             # Xavier Initialization for weights
#             size = layer.weight.size()
#             fan_out = size[0]
#             fan_in = size[1]
#             std = np.sqrt(2.0 / (fan_in + fan_out))
#             layer.weight.data.normal_(0.0, std)

#             # Normal Initialization for weights
#             layer.bias.data.normal_(0.0, 0.001)
        
#         for layer in self.out_layers:
#             # Xavier Initialization for weights
#             size = layer.weight.size()
#             fan_out = size[0]
#             fan_in = size[1]
#             std = np.sqrt(2.0 / (fan_in + fan_out))
#             layer.weight.data.normal_(0.0, std)

#             # Normal Initialization for weights
#             layer.bias.data.normal_(0.0, 0.001)
        
#         size = self.emb_layer.weight.size()
#         fan_out = size[0]
#         fan_in = size[1]
#         std = np.sqrt(2.0 / (fan_in + fan_out))
#         self.emb_layer.weight.data.normal_(0.0, std)
#         self.emb_layer.bias.data.normal_(0.0, 0.001)
    
#     def forward(self, x, timesteps):
#         time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
#         emb = self.emb_layer(time_emb)

#         if self.norm:
#             x = F.normalize(x)
#         x = self.drop(x)

#         if x.dim() ==3:
#             emb = emb.unsqueeze(1).expand(-1, x.size(1), -1) 
#         h = torch.cat([x, emb], dim=-1)
#         for i, layer in enumerate(self.in_layers):
#             h = layer(h)
#             h = torch.tanh(h)
        
#         for i, layer in enumerate(self.out_layers):
#             h = layer(h)
#             if i != len(self.out_layers) - 1:
#                 h = torch.tanh(h)
#         h = self.emb_out_linear(h)
#         return h
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class MLP_Denoise_Network(nn.Module):
    """
    简单的MLP降噪网络，支持动态输入维度
    """
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
        super(MLP_Denoise_Network, self).__init__()
        self.base_input_dim = in_dims[0]  # 基础输入维度
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm
        self.dropout_rate = dropout
        
        # 时间嵌入层
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
            SiLU(),
            nn.Linear(self.time_emb_dim * 4, self.time_emb_dim)
        )
        
        # 缓存不同输入维度的解码器
        self.cached_decoders = {}
        
        self.drop = nn.Dropout(dropout)
    
    def get_or_create_decoder(self, input_dim):
        """根据输入维度动态创建MLP解码器"""
        if input_dim not in self.cached_decoders:
            # 计算隐藏层大小
            hidden_size = input_dim + self.time_emb_dim
            
            # 创建MLP解码器
            decoder = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                SiLU(),
                nn.Linear(hidden_size * 4, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, input_dim)  # 输出层，映射回原始维度
            )
            
            # 初始化权重
            for module in decoder.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
            
            # 移动到正确的设备
            device = next(self.parameters()).device
            decoder = decoder.to(device)
            
            self.cached_decoders[input_dim] = decoder
        
        return self.cached_decoders[input_dim]
    
    def forward(self, x, timesteps):
        # 获取实际输入维度
        actual_input_dim = x.shape[-1]
        
        # 获取或创建对应维度的解码器
        decoder = self.get_or_create_decoder(actual_input_dim)
        
        # 时间嵌入
        time_emb = self.time_embed(timestep_embedding(timesteps, self.time_emb_dim)).to(x.device)
        
        # 归一化和dropout
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)
        
        # 处理不同维度的输入
        if x.dim() == 3:
            # [batch_size, seq_len, feature_dim]
            time_emb = time_emb.unsqueeze(1).expand(-1, x.size(1), -1)
        
        # 拼接输入和时间嵌入
        h = torch.cat([x, time_emb], dim=-1)
        
        # 通过MLP解码器
        output = decoder(h)
        
        return output

class DNN(nn.Module):
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
        super(DNN, self).__init__()
        self.base_in_dims = in_dims  # 保存基础维度配置
        self.base_out_dims = out_dims
        self.base_input_dim = in_dims[0]  # 基础输入维度 (100)
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm
        self.dropout_rate = dropout
        self.time_embed = nn.Sequential(nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
                                        SiLU(),
                                        nn.Linear(self.time_emb_dim * 4, self.time_emb_dim)
                                        )
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        
        # 缓存不同输入维度的网络层
        self.cached_networks = {}
        
        self.drop = nn.Dropout(dropout)
        self.init_emb_weights()
    
    def init_emb_weights(self):
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)
    
    def get_or_create_layers(self, input_dim):
        """根据输入维度动态创建网络层"""
        if input_dim not in self.cached_networks:
            # 计算缩放因子
            scale_factor = input_dim / self.base_input_dim
            
            # 缩放维度
            scaled_in_dims = [int(dim * scale_factor) if dim == self.base_input_dim else dim 
                             for dim in self.base_in_dims]
            scaled_out_dims = [int(dim * scale_factor) if dim == self.base_out_dims[-1] else dim 
                              for dim in self.base_out_dims]
            
            # 确保输入输出维度匹配
            scaled_in_dims[0] = input_dim
            scaled_out_dims[-1] = input_dim
            
            # 创建网络层
            if self.time_type == "cat":
                in_dims_temp = [scaled_in_dims[0] + self.time_emb_dim] + scaled_in_dims[1:]
            else:
                raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
            out_dims_temp = scaled_out_dims
            
            in_layers = nn.ModuleList([nn.Linear(d_in, d_out) 
                for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
            out_layers = nn.ModuleList([nn.Linear(d_in, d_out) 
                for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
            emb_out_linear = nn.Linear(scaled_out_dims[-1], input_dim)
            
            # 初始化权重
            for layer in in_layers:
                size = layer.weight.size()
                fan_out = size[0]
                fan_in = size[1]
                std = np.sqrt(2.0 / (fan_in + fan_out))
                layer.weight.data.normal_(0.0, std)
                layer.bias.data.normal_(0.0, 0.001)
            
            for layer in out_layers:
                size = layer.weight.size()
                fan_out = size[0]
                fan_in = size[1]
                std = np.sqrt(2.0 / (fan_in + fan_out))
                layer.weight.data.normal_(0.0, std)
                layer.bias.data.normal_(0.0, 0.001)
            
            size = emb_out_linear.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            emb_out_linear.weight.data.normal_(0.0, std)
            emb_out_linear.bias.data.normal_(0.0, 0.001)
            
            # 移动到正确的设备
            device = next(self.parameters()).device
            in_layers = in_layers.to(device)
            out_layers = out_layers.to(device)
            emb_out_linear = emb_out_linear.to(device)
            
            self.cached_networks[input_dim] = {
                'in_layers': in_layers,
                'out_layers': out_layers,
                'emb_out_linear': emb_out_linear
            }
        
        return self.cached_networks[input_dim]
    
    def forward(self, x, timesteps):
        # 获取实际输入维度
        actual_input_dim = x.shape[-1]
        
        # 获取或创建对应维度的网络层
        network = self.get_or_create_layers(actual_input_dim)
        in_layers = network['in_layers']
        out_layers = network['out_layers']
        emb_out_linear = network['emb_out_linear']
        
        # 时间嵌入
        time_emb = self.time_embed(timestep_embedding(timesteps, self.time_emb_dim)).to(x.device)
        emb = self.emb_layer(time_emb)

        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)

        if x.dim() == 3:
            emb = emb.unsqueeze(1).expand(-1, x.size(1), -1) 
        
        h = torch.cat([x, emb], dim=-1)
        
        # 通过 in_layers
        for i, layer in enumerate(in_layers):
            h = layer(h)
            h = torch.tanh(h)
        
        # 通过 out_layers
        for i, layer in enumerate(out_layers):
            h = layer(h)
            if i != len(out_layers) - 1:
                h = torch.tanh(h)
        
        # 输出层
        h = emb_out_linear(h)
        return h

class TransformerDNN(nn.Module):
    """
    A Transformer-based neural network for the reverse diffusion process.
    支持动态输入维度
    """
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=True, dropout=0.5, nhead=4, num_layers=1):
        super(TransformerDNN, self).__init__()
        self.base_in_dims = in_dims
        self.base_out_dims = out_dims
        self.base_input_dim = in_dims[0]  # 基础输入维度
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm
        self.dropout_rate = dropout
        self.nhead = nhead
        self.num_layers = num_layers

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        
        # 缓存不同输入维度的网络层
        self.cached_networks = {}
        
        self.drop = nn.Dropout(dropout)
        self.init_emb_weights()
    
    def init_emb_weights(self):
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)
    
    def get_or_create_layers(self, input_dim):
        """根据输入维度动态创建网络层"""
        if input_dim not in self.cached_networks:
            # 计算缩放因子
            scale_factor = input_dim / self.base_input_dim
            
            # 缩放维度
            scaled_in_dims = [int(dim * scale_factor) if dim == self.base_input_dim else dim 
                             for dim in self.base_in_dims]
            scaled_out_dims = [int(dim * scale_factor) if dim == self.base_out_dims[-1] else dim 
                              for dim in self.base_out_dims]
            
            # 确保输入输出维度匹配
            scaled_in_dims[0] = input_dim
            scaled_out_dims[-1] = input_dim
            
            # 创建网络层
            # 注意：时间嵌入维度是固定的 self.time_emb_dim，不需要缩放
            if self.time_type == "cat":
                # input_dim + time_emb_dim -> input_dim
                input_projection = nn.Linear(input_dim + self.time_emb_dim, input_dim)
            else:
                raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
            
            # 确保 input_dim 能被 nhead 整除
            if input_dim % self.nhead != 0:
                # 调整 nhead 为能整除 input_dim 的最大值
                adjusted_nhead = self.nhead
                while input_dim % adjusted_nhead != 0 and adjusted_nhead > 1:
                    adjusted_nhead -= 1
                if adjusted_nhead == 0:
                    adjusted_nhead = 1
            else:
                adjusted_nhead = self.nhead
            
            # Define Transformer Encoder Layers
            transformer_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=adjusted_nhead, dropout=self.dropout_rate)
            transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=self.num_layers)
            
            # 创建 out_layers
            out_dims_temp = scaled_out_dims
            out_layers = nn.ModuleList([nn.Linear(d_in, d_out) 
                for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
            
            # 投影层
            if input_dim != scaled_out_dims[0]:
                to_out_dims_proj = nn.Linear(input_dim, scaled_out_dims[0])
            else:
                to_out_dims_proj = None
            
            # 输出层
            emb_out_linear = nn.Linear(scaled_out_dims[-1], input_dim)
            
            # 初始化权重
            size = input_projection.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            input_projection.weight.data.normal_(0.0, std)
            input_projection.bias.data.normal_(0.0, 0.001)
            
            for layer in out_layers:
                size = layer.weight.size()
                fan_out = size[0]
                fan_in = size[1]
                std = np.sqrt(2.0 / (fan_in + fan_out))
                layer.weight.data.normal_(0.0, std)
                layer.bias.data.normal_(0.0, 0.001)
            
            if to_out_dims_proj is not None:
                size = to_out_dims_proj.weight.size()
                fan_out = size[0]
                fan_in = size[1]
                std = np.sqrt(2.0 / (fan_in + fan_out))
                to_out_dims_proj.weight.data.normal_(0.0, std)
                to_out_dims_proj.bias.data.normal_(0.0, 0.001)
            
            size = emb_out_linear.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            emb_out_linear.weight.data.normal_(0.0, std)
            emb_out_linear.bias.data.normal_(0.0, 0.001)
            
            # 移动到正确的设备
            device = next(self.parameters()).device
            input_projection = input_projection.to(device)
            transformer_encoder = transformer_encoder.to(device)
            out_layers = out_layers.to(device)
            if to_out_dims_proj is not None:
                to_out_dims_proj = to_out_dims_proj.to(device)
            emb_out_linear = emb_out_linear.to(device)
            
            self.cached_networks[input_dim] = {
                'input_projection': input_projection,
                'transformer_encoder': transformer_encoder,
                'out_layers': out_layers,
                'to_out_dims_proj': to_out_dims_proj,
                'emb_out_linear': emb_out_linear
            }
        
        return self.cached_networks[input_dim]
    
    def forward(self, x, timesteps):
        # 获取实际输入维度
        actual_input_dim = x.shape[-1]
        
        # 获取或创建对应维度的网络层
        network = self.get_or_create_layers(actual_input_dim)
        input_projection = network['input_projection']
        transformer_encoder = network['transformer_encoder']
        out_layers = network['out_layers']
        to_out_dims_proj = network['to_out_dims_proj']
        emb_out_linear = network['emb_out_linear']
        
        # 时间嵌入
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        
        # Normalize input if specified
        if self.norm:
            x = F.normalize(x)
        
        # Apply dropout to input
        x = self.drop(x)

        if x.dim() == 3:
            emb = emb.unsqueeze(1).expand(-1, x.size(1), -1)
        elif x.dim() == 2: 
            emb = emb

        h = torch.cat([x, emb], dim=-1)
        
        h = input_projection(h)
        
        # Transformer encoder processing
        if h.dim() == 3:
            h = transformer_encoder(h)
        elif h.dim() == 2:
            h = h.unsqueeze(1)  # Add sequence dimension
            h = transformer_encoder(h)
            h = h.squeeze(1)  # Remove sequence dimension
        
        # 通过投影层
        if to_out_dims_proj is not None:
            h = to_out_dims_proj(h)
        
        # 通过 out_layers
        for i, layer in enumerate(out_layers):
            h = layer(h)
            if i != len(out_layers) - 1:
                h = torch.tanh(h)
        
        # 输出层
        h = emb_out_linear(h)
        
        return h

class UNet(nn.Module):
    """
    UNet 降噪网络，支持动态输入维度
    """
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
        super(UNet, self).__init__()
        self.base_in_dims = in_dims
        self.base_out_dims = out_dims
        self.base_input_dim = in_dims[0]  # 基础输入维度
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm
        self.dropout_rate = dropout

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        
        # 缓存不同输入维度的网络层
        self.cached_networks = {}
        
        self.drop = nn.Dropout(dropout)
        self.init_emb_weights()

    def init_emb_weights(self):
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)
    
    def get_or_create_layers(self, input_dim):
        """根据输入维度动态创建网络层"""
        if input_dim not in self.cached_networks:
            # 计算缩放因子
            scale_factor = input_dim / self.base_input_dim
            
            # 根据输入维度调整隐藏层维度
            # 原始: input_dim=100 -> 64, 128, 256
            # 拆分: input_dim=20 -> 13, 26, 51 (约为原始的 1/5)
            hidden_dim_1 = max(8, int(64 * scale_factor))  # 至少 8
            hidden_dim_2 = max(16, int(128 * scale_factor))  # 至少 16
            hidden_dim_3 = max(32, int(256 * scale_factor))  # 至少 32
            
            # 创建网络层
            if self.time_type == "cat":
                in_dim = input_dim + self.time_emb_dim
            else:
                raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)

            # Encoder (downsampling)
            enc1 = nn.Linear(in_dim, hidden_dim_1)
            enc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
            enc3 = nn.Linear(hidden_dim_2, hidden_dim_3)

            # Decoder (upsampling)
            dec3 = nn.Linear(hidden_dim_3, hidden_dim_2)
            dec2 = nn.Linear(hidden_dim_2 * 2, hidden_dim_1)  # *2 because of skip connection
            dec1 = nn.Linear(hidden_dim_1 * 2, input_dim)  # *2 because of skip connection, 输出维度为 input_dim

            # 初始化权重
            for layer in [enc1, enc2, enc3, dec3, dec2, dec1]:
                nn.init.xavier_normal_(layer.weight)
                nn.init.normal_(layer.bias, std=1e-6)
            
            # 移动到正确的设备
            device = next(self.parameters()).device
            enc1 = enc1.to(device)
            enc2 = enc2.to(device)
            enc3 = enc3.to(device)
            dec3 = dec3.to(device)
            dec2 = dec2.to(device)
            dec1 = dec1.to(device)
            
            self.cached_networks[input_dim] = {
                'enc1': enc1,
                'enc2': enc2,
                'enc3': enc3,
                'dec3': dec3,
                'dec2': dec2,
                'dec1': dec1
            }
        
        return self.cached_networks[input_dim]

    def forward(self, x, timesteps):
        # 获取实际输入维度
        actual_input_dim = x.shape[-1]
        
        # 获取或创建对应维度的网络层
        network = self.get_or_create_layers(actual_input_dim)
        enc1 = network['enc1']
        enc2 = network['enc2']
        enc3 = network['enc3']
        dec3 = network['dec3']
        dec2 = network['dec2']
        dec1 = network['dec1']
        
        # 时间嵌入
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)

        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)

        if x.dim() == 3:
            emb = emb.unsqueeze(1).expand(-1, x.size(1), -1)
        h = torch.cat([x, emb], dim=-1)

        # Encoder
        e1 = F.relu(enc1(h))
        e2 = F.relu(enc2(e1))
        e3 = F.relu(enc3(e2))

        # Decoder with skip connections
        d3 = F.relu(dec3(e3))
        d2 = F.relu(dec2(torch.cat([d3, e2], dim=-1)))
        d1 = dec1(torch.cat([d2, e1], dim=-1))

        return d1






class GatingNetwork(nn.Module):
    def __init__(self, in_features,experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(in_features, experts)
        self.softmax = nn.Softmax(dim=-1)  # 使用dim=-1以支持2维和3维

    def forward(self, x):
        # 支持2维和3维输入
        # 2维: [batch_size, feature_dim] -> [batch_size, experts]
        # 3维: [batch_size, seq_len, feature_dim] -> [batch_size, seq_len, experts]
        x = self.fc(x)
        x = self.softmax(x)
        return x

class router(nn.Module):
    def __init__(self, dim, channel_num, t, prefer_idx=2, prefer_bias=0.0):
        super().__init__()
        self.l1 = nn.Linear(dim, int(dim/8))
        self.l2 = nn.Linear(int(dim/8), channel_num)
        self.t = t
        self.prefer_idx = prefer_idx
        self.prefer_bias = prefer_bias  # 给某个专家额外加 logit 偏置

    def forward(self, x):
        # x: [B, D]
        h = F.relu(F.normalize(self.l1(x), p=2, dim=1))
        logits = self.l2(h) / self.t
        if self.prefer_bias != 0:
            logits[:, self.prefer_idx] = logits[:, self.prefer_idx] + self.prefer_bias
        return torch.softmax(logits, dim=1)

# import torch
# import torch.nn as nn

# class ExpertAttentionFusion(nn.Module):
#     def __init__(self, dim, num_heads=4, dropout=0.1):
#         super().__init__()
#         # batch_first=True 确保输入格式为 [batch, seq_len, feature]
#         self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, 
#                                                dropout=dropout, batch_first=True)
        
#         # 关键：为每个专家学习一个独特的身份向量
#         # 形状: [1, 3, dim] -> 对应 3 个专家
#         self.expert_embedding = nn.Parameter(torch.randn(1, 3, dim))
        
#         self.norm = nn.LayerNorm(dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, e1, e2, e3, weights=None):
#         """
#         e1, e2, e3: 形状均为 [Batch, Dim] (e.g., [50, 100])
#         weights: 可选，来自 Router 的门控权重 [Batch, 3]
#         """
#         # 1. 堆叠: [Batch, Dim] -> [Batch, 3, Dim]
#         # stack 将三个专家的输出在 dim=1 维度拼接，形成序列
#         expert_seq = torch.stack([e1, e2, e3], dim=1)
        
#         # 2. 加入专家身份编码 (Broadcasting)
#         # 让模型知道 position 0 是 DNN, position 1 是 Transformer...
#         expert_seq = expert_seq + self.expert_embedding
        
#         # 3. 自注意力交互
#         # attn_output 形状: [Batch, 3, Dim]
#         attn_output, _ = self.self_attn(expert_seq, expert_seq, expert_seq)
        
#         # 残差连接 + Norm (标准 Transformer 操作)
#         expert_seq = self.norm(expert_seq + self.dropout(attn_output))
        
#         # 4. 最终融合 (两种策略)
        
#         if weights is not None:
#             # 策略 A: 结合你现有的 Router 权重进行加权求和 (MoE 风格)
#             # weights 形状: [Batch, 3] -> [Batch, 3, 1]
#             weights = weights.unsqueeze(-1)
#             # 加权求和: sum([Batch, 3, Dim] * [Batch, 3, 1], dim=1) -> [Batch, Dim]
#             final_output = (expert_seq * weights).sum(dim=1)
#         else:
#             # 策略 B: 直接平均池化 (如果没有 Router 权重)
#             # [Batch, 3, Dim] -> [Batch, Dim]
#             final_output = expert_seq.mean(dim=1)
            
#         return final_output

# class MoE_Denoise_Network(nn.Module):
#     def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
#         super(MoE_Denoise_Network, self).__init__()

#         self.gating_network = GatingNetwork(in_dims[0], experts=3)
#         self.router = router(dim=in_dims[0], channel_num=3, t=0.7, prefer_idx=2, prefer_bias=2)
        
#         self.expert_dnn = DNN(in_dims, out_dims, emb_size, time_type="cat", norm=norm, dropout=dropout)
#         self.expert_transformer_dnn = MLP_Denoise_Network(in_dims, out_dims, emb_size, time_type="cat", norm=norm, dropout=dropout)
#         self.expert_unet = UNet(in_dims, out_dims, emb_size, time_type="cat", norm=norm, dropout=dropout)
        
#         # === 新增：注意力融合层 ===
#         # 假设输出维度 out_dims 是 100，这里需要传入这个维度
#         # 如果 out_dims 是列表，取 out_dims[-1] 或具体的 feature dim
#         feature_dim = 100 # 请替换为你实际的 out_dims 数值
#         self.attn_fusion = ExpertAttentionFusion(dim=out_dims[-1], num_heads=4)

#     def forward(self, x, timesteps):
#         # x: [50, 100]
        
#         # 1. 获取 Router 权重 (用于后续加权)
#         # gating_scores: [50, 3] (假设 router 输出 softmax 后的概率)
#         gating_scores = self.router(x) 
        
#         # 2. 获取三个专家的输出
#         # 全部为 [50, 100]
#         out_dnn = self.expert_dnn(x, timesteps)
#         out_trans = self.expert_transformer_dnn(x, timesteps)
#         out_unet = self.expert_unet(x, timesteps)
        
#         # 3. 注意力融合
#         # 输入三个专家输出 + Router 权重
#         # 输出: [50, 100]
#         final_output = self.attn_fusion(out_dnn, out_trans, out_unet, weights=gating_scores)
        
#         return final_output

import torch
import torch.nn as nn

class UNetGuidedFusion(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        # Cross Attention: Query=UNet, Key/Value=Others
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, 
                                                dropout=dropout, batch_first=True)
        
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
        # === 核心稳健设计：零初始化投影层 ===
        # 作用：训练初始阶段输出全为0，保证模型等价于单体UNet，避免"冷启动破坏"
        self.output_proj = nn.Linear(dim, dim)
        self._init_zero_weights()

    def _init_zero_weights(self):
        nn.init.constant_(self.output_proj.weight, 0)
        nn.init.constant_(self.output_proj.bias, 0)

    def forward(self, out_dnn, x_start):
        """
        out_dnn: DNN输出 (Key/Value)
        x_start: 原始输入 (Query)
        """
        original_shape = out_dnn.shape
        feat_dim = out_dnn.shape[-1]
    
        query = out_dnn.reshape(-1, 1, feat_dim) # [N, 1, D]
        k = x_start.reshape(-1, 1, feat_dim)    # [N, 1, D]
        v = x_start.reshape(-1, 1, feat_dim)    # [N, 1, D]
        attn_out, _ = self.cross_attn(query=query, key=k, value=v)
        output = self.norm(self.dropout(attn_out))
        output = self.output_proj(output)
        output = output.reshape(original_shape)
        
        return output

class MoE_Denoise_Network(nn.Module):
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
        super(MoE_Denoise_Network, self).__init__()
        self.gating_network = GatingNetwork(in_dims[0], experts=3)
        self.expert_dnn = DNN(in_dims, out_dims, emb_size, time_type="cat", norm=norm, dropout=dropout)
        self.expert_transformer_dnn = MLP_Denoise_Network(in_dims, out_dims, emb_size, time_type="cat", norm=norm, dropout=dropout)
        self.expert_unet = UNet(in_dims, out_dims, emb_size, time_type="cat", norm=norm, dropout=dropout)

    def forward(self, x, timesteps):
        # 直接使用原始输入，保持维度不变
        # 2维: [batch_size, feature_dim]
        # 3维: [batch_size, seq_len, feature_dim]
        is_3d = x.dim() == 3
        
        gating_scores = self.gating_network(x)
        
        # Get outputs from all experts
        output_dnn = self.expert_dnn(x, timesteps)
        output_transformer_dnn = self.expert_transformer_dnn(x, timesteps)
        output_unet = self.expert_unet(x, timesteps)
        
        # Weighted combination: maintain input dimensionality
        if is_3d:
            output = (output_dnn * gating_scores[:, :, 0:1] + 
                     output_transformer_dnn * gating_scores[:, :, 1:2] + 
                     output_unet * gating_scores[:, :, 2:3])
        else:
            output = (output_dnn * gating_scores[:, 0:1] + 
                     output_transformer_dnn * gating_scores[:, 1:2] + 
                     output_unet * gating_scores[:, 2:3])
        # output = 0.3*output_dnn+0.3*output_transformer_dnn+0.4*output_unet
        return output