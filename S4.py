import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(42)

def random_SSM(N, D):
    A = torch.randn(N, N)
    B = torch.randn(N, D)
    C = torch.randn(D, N)
    return A, B, C

def zero_order_discretize(A, B, C, step):
    Ab = torch.matrix_exp(A * step)
    I = torch.eye(A.size(0))
    Bb = torch.linalg.solve(A, (Ab - I) @ B)
    return Ab, Bb, C

def bilinear_discretize(A, B, C, step):
    I = torch.eye(A.size(0))
    BL = torch.linalg.inv(I - A * (step / 2.0))
    Ab = BL @ (I + A * (step / 2.0))
    Bb = (BL * step) @ B
    return Ab, Bb, C

def scan(Ab, Bb, Cb, u, x0):
    x = x0
    ys = []
    for u_k in u:
        x = Ab @ x + Bb @ u_k
        y = Cb @ x
        ys.append(y)
    return x, torch.stack(ys)

def run_SSM(A, B, C, u, discretization='bilinear'):
    L = u.shape[0]
    N = A.shape[0]
    if discretization == 'zero_order':
        Ab, Bb, Cb = zero_order_discretize(A, B, C, 1.0 / L)
    elif discretization == 'bilinear':
        Ab, Bb, Cb = bilinear_discretize(A, B, C, 1.0 / L)
    return scan(Ab, Bb, Cb, u, torch.zeros(N))
    
def k_conv(Ab, Bb, Cb, L):
    k_list = []
    current_Ab_power = torch.eye(Ab.shape[0], device=Ab.device, dtype=Ab.dtype)

    for _ in range(L):
        k = Cb @ current_Ab_power @ Bb
        k_list.append(k.flatten())
        current_Ab_power = current_Ab_power @ Ab

    return torch.stack(k_list)
    
def causal_convolution(u, K, nofft=False):
    L = u.shape[0]

    if nofft:
        K_flipped = torch.flip(K, dims=[-1])

        u_reshaped = u.unsqueeze(0).unsqueeze(0)
        K_reshaped = K_flipped.unsqueeze(0).unsqueeze(0)

        padding = K.shape[0] - 1
        full_conv = F.conv1d(u_reshaped, K_reshaped, padding=padding)

        return full_conv.squeeze()[:L]
    else:
        assert K.shape[0] == L

        fft_size = 2 * L

        pad_dims = (0, 0, 0, L)
        u_padded = F.pad(u, pad_dims)
        K_padded = F.pad(K, pad_dims)
        
        u_d = torch.fft.rfft(u_padded, n=fft_size, dim=0)
        K_d = torch.fft.rfft(K_padded, n=fft_size, dim=0)

        out_d = u_d * K_d
        
        return torch.fft.irfft(out_d, n=fft_size, dim=0)[:L]

def test_cnn_is_rnn(N=4, L=16, step = 1.0 / 16):
    A, B, C = random_SSM(N, 1)
    u = torch.rand(L).unsqueeze(1)
    _, rec = run_SSM(A, B, C, u, discretization='bilinear')

    Ab, Bb, Cb = bilinear_discretize(A, B, C, step)
    conv = causal_convolution(u, k_conv(Ab, Bb, Cb, L))

    assert torch.allclose(rec, conv, atol=1e-5), "CNN and RNN outputs do not match!"

def log_step_initiallizer(dt_min=0.001, dt_max=0.1):
    def init(shape):
        return torch.rand(shape) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
    return init

def lecun_normal_init(tensor):
    fan_in = nn.init._calculate_fan_in_and_fan_out(tensor)[0]
    if fan_in != 0:
        std = math.sqrt(1.0 / fan_in)
        with torch.no_grad():
            return tensor.normal_(0, std)
    else:
        return tensor

class SSMLayer(nn.Module):
    def __init__(self, N, l_max, decode=False):
        super()(SSMLayer, self).__init__()
        self.N = N
        self.l_max = l_max
        self.decode = decode
        self.A = nn.Parameter(lecun_normal_init(torch.empty(N, N)))
        self.B = nn.Parameter(lecun_normal_init(torch.empty(N, 1)))
        self.C = nn.Parameter(lecun_normal_init(torch.empty(1, N)))
        self.D = nn.Parameter(torch.ones(1))
        log_step_initiallizer = log_step_initiallizer()
        self.log_step = nn.Parameter(log_step_initiallizer((1,)))

        self.step = torch.exp(self.log_step)
        self.Ab, self.Bb, self.Cb = bilinear_discretize(self.A, self.B, self.C, self.step)
        self.K = k_conv(self.Ab, self.Bb, self.Cb, l_max)

        self.register_buffer('x_k_1', torch.zeros(1, N))
    
    def forward(self, u):
        if not self.decode:
            return causal_convolution(u, self.K) + self.D * u
        else:
            x_k, y_s = scan(self.Ab, self.Bb, self.Cb, u, self.x_k_1)
            self.x_k_1 = x_k.detach()
            return y_s + self.D * u
    
class SequenceBlock(nn.Module):

    def __init__(self, layer_cls, layer_args, dropout, d_model, prenorm=True, glu=True, decode=False):
        super(SequenceBlock, self).__init__()
        self.d_model = d_model
        self.prenorm = prenorm
        self.glu = glu

        self.seq = layer_cls(**layer_args, decode=decode)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if self.glu:
            self.out1 = nn.Linear(d_model, d_model)
            self.out2 = nn.Linear(d_model, d_model)
        else:
            self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        skip = x
        if self.prenorm:
            x = self.norm(x)
        x = self.seq(x)
        x = self.dropout(F.gelu(x))
        if self.glu:
            x = self.out1(x) * torch.sigmoid(self.out2(x))
        else:
            x = self.out(x)
        x = skip + self.dropout(x)
        if not self.prenorm:
            x = self.norm(x)
        return x

def example_mass(k, b, m):
    A = torch.tensor([[0, 1], [-k/m, -b/m]], dtype=torch.float32)
    B = torch.tensor([[0], [1/m]], dtype=torch.float32)
    C = torch.tensor([[1, 0]], dtype=torch.float32)
    return A, B, C

def example_force(t):
    x = torch.sin(10 * t)
    return x * (x > 0.5)

def example_SSM():
    A, B, C = example_mass(40, 5, 1)
    L = 100
    step = 1.0 / L
    ks = torch.arange(L)
    u = example_force(ks * step).unsqueeze(1)
    _, y = run_SSM(A, B, C, u, discretization='bilinear')
    import matplotlib.pyplot as plt
    import seaborn
    from celluloid import Camera

    seaborn.set_context("paper")
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    camera = Camera(fig)
    ax1.set_title("Force $u_k$")
    ax2.set_title("Position $y_k$")
    ax3.set_title("Object")
    ax1.set_xticks([], [])
    ax2.set_xticks([], [])

    # Animate plot over time
    for k in range(0, L, 2):
        ax1.plot(ks[:k], u[:k], color="red")
        ax2.plot(ks[:k], y[:k], color="blue")
        ax3.boxplot(
            [[y[k, 0] - 0.04, y[k, 0], y[k, 0] + 0.04]],
            showcaps=False,
            whis=False,
            vert=False,
            widths=10,
        )
        camera.snap()
    anim = camera.animate()
    anim.save("images/line.gif", dpi=150, writer="imagemagick")

def main():
    test_cnn_is_rnn()

if __name__ == "__main__":
    main()