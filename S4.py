import torch
import torch.nn as nn
import torch.nn.functional as F

def random_SSM(self, seed, N, D):
    torch.manual_seed(seed)
    A = torch.randn(N, N)
    B = torch.randn(N, D)
    C = torch.randn(D, N)
    return A, B, C

class S4(nn.Module):
    def __init__(self, A, B, C, N, D, seed=0):
        super(S4, self).__init__()
        self.A = A
        self.B = B
        self.C = C
        if A is None or B is None or C is None:
            A, B, C = random_SSM(self, seed, N, D)
        self.N = N
        self.D = D
        self.seed = seed

    def zero_order_discretize(self, A, B, C, step):
        Ab = torch.exp(A * step)
        Bb = torch.linalg.inv(A * step) @ (Ab - torch.eye(A.size(0))) * step @ B
        return Ab, Bb, C

    def bilinear_discretize(self, A, B, C, step):
        I = torch.eye(A.size(0))
        BL = torch.linalg.inv(I - A * (step / 2.0))
        Ab = BL @ (I + A * (step / 2.0))
        Bb = (BL * step) @ B
        return Ab, Bb, C

    def scan(self, u, x0):
        x = x0
        ys = []
        for u_k in u:
            x = self.Ab @ x + self.Bb @ u_k
            y = self.Cb @ x
            ys.append(y)
        return torch.stack(ys)

    def run_SSM(self, u, discretization='bilinear'):
        L = u.shape[0]
        N = self.A.shape[0]
        if discretization == 'zero_order':
            self.Ab, self.Bb, self.Cb = self.zero_order_discretize(self.A, self.B, self.C, 1.0 / L)
        elif discretization == 'bilinear':
            self.Ab, self.Bb, self.Cb = self.bilinear_discretize(self.A, self.B, self.C, 1.0 / L)
        return self.scan(u, torch.zeros(N))

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
    ssm = S4(A, B, C, N=2, D=1)
    y = ssm.run_SSM(u, discretization='bilinear')
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
    example_SSM()

if __name__ == "__main__":
    main()