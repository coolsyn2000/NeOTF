import numpy as np
import torch
import torch.nn as nn
from thop import profile

class SIREN(nn.Module):
    def __init__(self, omega_0=30.0):
        super(SIREN, self).__init__()
        self.omega_0 = omega_0

    def forward(self, x):
        return torch.sin(self.omega_0 * x)


def fourier_encode(coords, in_features, num_frequencies):
    """
    对输入坐标进行傅里叶编码
    :param coords: 输入坐标，形状为 (batch_size, num_points, 2)
    :param num_frequencies: 傅里叶频率的数量
    :return: 编码后的坐标，形状为 (batch_size, num_points, 2 * num_frequencies)
    """
    num_points, _ = coords.shape
    encoded = []

    for freq in range(num_frequencies):
        frequency = 2 ** freq  # 使用 2 的幂作为频率
        encoded.append(torch.sin(frequency * coords[..., 0]))  # 正弦
        encoded.append(torch.cos(frequency * coords[..., 0]))
        encoded.append(torch.sin(frequency * coords[..., 1]))  # 正弦
        encoded.append(torch.cos(frequency * coords[..., 1]))
        # 余弦

    return torch.stack(encoded, dim=-1).reshape(num_points, -1)  # 在最后一个维度拼接


class SineLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30, num_frequencies=10):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.in_features = num_frequencies*4
        self.net = []
        self.net.append(SineLayer(self.in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
            self.net.append(nn.Tanh())
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input

        encoded_coords = fourier_encode(coords, self.in_features, self.num_frequencies)

        output = self.net(encoded_coords) * torch.pi
        return output, coords
    
if __name__ == "__main__":
    model = Siren(in_features=2, out_features=1, hidden_features=128, hidden_layers=2, outermost_linear=True,
                  first_omega_0=30, hidden_omega_0=30, num_frequencies=8)
    coords = torch.rand((int(512*512*0.007), 2))
    output, coords_out = model(coords)

    macs, params = profile(model, inputs=(coords,))

    # 3. 使用clever_format进行格式化输出
    # thop算出的macs，乘以2得到GFLOPs
    gflops = macs * 2
    print(f"MACs: {macs}, Params: {params}, GFLOPs: {gflops / 1e9}")
    print(output.shape)  # 应该输出 torch.Size([100, 1])
    print(coords_out.shape)  # 应该输出 torch.Size([100, 2])
