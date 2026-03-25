import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)


class PINN(nn.Module):
    def __init__(self, hidden_layers=4, neurons=32):
        super().__init__()
        layers = [nn.Linear(1, neurons), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(neurons, neurons), nn.Tanh()]
        layers.append(nn.Linear(neurons, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def forcing(x):
    return -(np.pi ** 2) * torch.sin(np.pi * x)


def analytical(x):
    return torch.sin(np.pi * x)


def pde_residual(model, x):
    x = x.requires_grad_(True)
    u = model(x)
    u_x = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0]
    return u_xx - forcing(x)


def train():
    model = PINN(hidden_layers=4, neurons=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

    x_col = torch.linspace(0, 1, 100).unsqueeze(1)
    x_bc = torch.tensor([[0.0], [1.0]])
    u_bc = torch.tensor([[0.0], [0.0]])

    loss_history, loss_bc_history, loss_pde_history = [], [], []

    for epoch in range(1, 8001):
        optimizer.zero_grad()

        loss_bc = nn.MSELoss()(model(x_bc), u_bc)
        loss_pde = torch.mean(pde_residual(model, x_col) ** 2)
        loss = 10.0 * loss_bc + loss_pde

        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())
        loss_bc_history.append(loss_bc.item())
        loss_pde_history.append(loss_pde.item())

        if epoch % 1000 == 0:
            print(f"Epoch {epoch:5d} | Total: {loss.item():.2e} | BC: {loss_bc.item():.2e} | PDE: {loss_pde.item():.2e}")

    return model, loss_history, loss_bc_history, loss_pde_history


model, loss_hist, loss_bc_hist, loss_pde_hist = train()

x_test = torch.linspace(0, 1, 500).unsqueeze(1)
with torch.no_grad():
    u_pred = model(x_test)

u_exact = analytical(x_test)
l2_error = torch.sqrt(torch.mean((u_pred - u_exact) ** 2)).item()
print(f"\nL2 Error: {l2_error:.2e}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(x_test.numpy(), u_exact.numpy(), 'k-', lw=2, label='Analytical: sin(πx)')
axes[0].plot(x_test.numpy(), u_pred.numpy(), 'r--', lw=2, label='PINN')
axes[0].set_xlabel('x')
axes[0].set_ylabel('u(x)')
axes[0].set_title('PINN vs Analytical Solution')
axes[0].legend()
axes[0].grid(True)

error = torch.abs(u_pred - u_exact).numpy()
axes[1].plot(x_test.numpy(), error, 'b-', lw=1.5)
axes[1].set_xlabel('x')
axes[1].set_ylabel('|u_PINN - u_exact|')
axes[1].set_title(f'Pointwise Error (L2 = {l2_error:.2e})')
axes[1].grid(True)

epochs_range = range(1, len(loss_hist) + 1)
axes[2].semilogy(epochs_range, loss_hist,     label='Total Loss')
axes[2].semilogy(epochs_range, loss_bc_hist,  label='BC Loss')
axes[2].semilogy(epochs_range, loss_pde_hist, label='PDE Loss')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Loss (log scale)')
axes[2].set_title('Training Loss Curves')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig('pinn_bvp_solution.png', dpi=150, bbox_inches='tight')
plt.show()
