import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model.duo_hgp import SparseGP

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.integrate import solve_ivp

class WindyPendulum():
    def __init__(self, mass, gravity, length, friction, wind_res, wind_speed):
        self.m = mass
        self.g = gravity
        self.l = length
        self.r = friction
        self.k = wind_res
        self.v = wind_speed
        self.S = np.array([[0, 1], [-1, 0]])
    
    def hamiltonian(self, t, coords):
        q, p = coords[0], coords[1]
        d = np.exp((self.r - self.k * self.l**2) * t / (self.m * self.l**2))
        H = (p**2 / (2 * d * self.m * self.l**2) +
             d * self.m * self.g * self.l * (1 - np.cos(q)) +
             d * self.k * self.v * self.l * t * q)
        return H
    
    def equations_of_motion(self, t, x):
        q, p = x
        d = np.exp((self.r - self.k * self.l**2) * t / (self.m * self.l**2))

        # Partial derivatives of the Hamiltonian
        dH_dq = d * self.m * self.g * self.l * np.sin(q) + d * self.k * self.v * self.l * t
        dH_dp = p / (d * self.m * self.l**2)

        # Hamiltonian equations of motion
        dq_dt = dH_dp  # q' = ∂H/∂p
        dp_dt = -dH_dq  # p' = -∂H/∂q

        return [dq_dt, dp_dt]



    


class WindyPendulum1DTrajectoryDataset(Dataset):
    def __init__(self, num_samples=1000, time_steps=20, random_time=True, random_init=False, test=False):
        self.num_samples = num_samples
        self.time_steps = time_steps
        mass = 1.0
        gravity = 9.81
        length = 1.0
        friction = 0.2
        wind_res = 0.3
        wind_speed = 2
        T = 5.0
        timescale = 2.0
        samples = 100
        sigma = 0.1
        q_lower = -torch.pi/4
        q_upper = torch.pi/4
        self.system = WindyPendulum(mass, gravity, length, friction, wind_res, wind_speed)
        self.samples = samples
        self.T = T
        self.timescale = timescale
        self.time_off_scale = 0.0
        self.q_lower = q_lower
        self.q_upper = q_upper
        self.sigma = sigma
        self.random_time = random_time
        self.random_init = random_init
        self.test = test
        self.data = self.generate_trajectory()
    
    def get_initial_conditions(self):
        if self.random_init:
            q_values = np.linspace(self.q_lower, self.q_upper, self.num_samples)
        else:
            q_values = [self.q_lower] * self.num_samples
        p_values = np.zeros_like(q_values)
        initial_conditions = np.vstack([q_values, p_values]).T
        return initial_conditions



    def generate_trajectory(self):
        
        x0s = self.get_initial_conditions()
        
        trajectories = []
        for x0 in x0s:
            if self.random_time:
                t = np.sort(self.time_off_scale + np.random.rand(self.time_steps) * self.timescale)
            else:
                t = np.linspace(0, self.T, self.time_steps)
            sol = solve_ivp(self.system.equations_of_motion, [0, self.T], x0, t_eval=t, method='RK45', atol=1e-8, rtol=1e-8)
            qps = torch.tensor(sol.y.T, dtype=torch.float32)
            if not self.test:
                qps = qps + self.sigma * torch.randn_like(qps)
            ts = torch.tensor(np.stack([t], axis=1), dtype=torch.float32)
            trajectories.append((qps[:,0][:,None], ts))

        return trajectories    

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

# Custom collate function to separate time and trajectory data
def collate_fn(batch):
    t_batch, qp_batch = zip(*batch)  # Separate time and (x, y)
    t_batch = torch.stack(t_batch)  # Shape: (batch_size, time_steps)
    qp_batch = torch.stack(qp_batch)  # Shape: (batch_size, time_steps, 2d)
    return t_batch, qp_batch

# Create dataset and dataloader
dataset = WindyPendulum1DTrajectoryDataset(num_samples=100, time_steps=50)
dataloader = DataLoader(dataset, batch_size=100, collate_fn=collate_fn, shuffle=True)

test_dataset = WindyPendulum1DTrajectoryDataset(num_samples=100, time_steps=50, random_time=False, random_init=True, test=True)
test_dataloader = DataLoader(test_dataset, batch_size=100, collate_fn=collate_fn, shuffle=True)

device = "cuda:0"
input_dim = 2
num_inducing_points = 10
num_latents = 2
num_outputs = 2  # Number of labels/GPs
model = SparseGP(input_dim, num_inducing_points, num_latents, sigma_y=0.1).to(device)

# Optimizer
kernel_vars = [p for name, p in model.named_parameters() if 'kernel' in name]
encoding_vars = [p for name, p in model.named_parameters() if 'kernel' not in name]


from model.optimizers import SGLD, SGHMC

optimizer = torch.optim.Adam([
    {'params': kernel_vars, 'lr': 1e-2},
    {'params': encoding_vars, 'lr': 1e-2}
    ],
    lr=1e-2)
# optimizer = SGHMC(
#     model.parameters(),
#     num_burn_in_steps=0,
#     mdecay=0.9,
#     lr=1e-2)
# optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01)

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)

# Training loop
num_epochs = 100
model.train()



def plot_check(t, x):
    
    # We predict on test time horizon [1, 1.2] with 20 time steps
    test_time_horizon = np.linspace(0, 2.2, 100)
    # label is the label for the stochastic process we predict
    with torch.no_grad():
        means, covars, _, _, _, S_ms = model(torch.tensor(test_time_horizon).float().to(device).unsqueeze(1))
        Sm = S_ms.squeeze().detach().cpu().numpy()
        prior_mean = model.variational_mean.squeeze().detach().cpu().numpy()
        prior_covar = model.variational_covar.squeeze().detach().cpu().numpy()
        mean = means.squeeze().detach().cpu().numpy()
        covar = covars.squeeze().detach().cpu().numpy()

    std = np.sqrt(np.diag(covar)).reshape(-1)
    std_prior = np.sqrt(np.diag(prior_covar)).reshape(-1)
    #print(std)


    # We plot all the time series on the horizon [0, 1]
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 8), sharey= True)
    for i in range(min(t.shape[0],3)):
        ax1.plot(t[i], x[i], 'o-')
    ax1.set_title('Observation Samples')

    # Then we plot our prediction with uncertainty on test time horizon = [1, 1.2]
    ax2.plot(test_time_horizon, mean, c='red', lw=2, zorder=1, label='Mean prediction')
    ax2.fill_between(test_time_horizon, mean+2*std, mean-2*std, color='red', alpha=0.5, zorder=1)
    ax2.scatter(Sm, prior_mean, c='green', s=45, label='most informative timestamps')
    ax2.errorbar(Sm, prior_mean, yerr=2*std_prior, ls=' ')
    # Plot setting and plot show
    handle_list, _ = ax2.get_legend_handles_labels()
    handle_list.append(mpatches.Patch(color='red', label='Uncertainty region'))
    ax2.legend(handles=handle_list, fontsize='10', loc ="lower left")
    #M = 1.1*np.max(np.abs(Y_test))
    #plt.ylim(-M, M)
    ax2.set_title('Time series forecasting')
    return fig

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./logs/debug_optim/LBFGS')

model = model.to(device)
num_iters = 100
epoch = 0 
for _ in range(num_iters):
    for _, data in enumerate(dataloader):
        x, t = data
        optimizer.zero_grad()
        # Compute loss
        loss, logdet, quadratic, trace = model.compute_vfe_loss(t.to(device), x.to(device))
        print(loss.item(), logdet.item())
        # Backward pass
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()
        
        
        #optimizer.step(closure)

        if (epoch + 1) % 1 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}], Loss: {loss.item():.10f}")
            writer.add_scalar('training_loss', loss, epoch)
            writer.add_scalar('training_loss_logdet', logdet, epoch)
            writer.add_scalar('training_loss_quadratic', quadratic, epoch)
            writer.add_scalar('training_loss_trace', trace, epoch)
            writer.add_figure('check process', plot_check(t, x), global_step=epoch)
            test_loss = 0.0
            cnt = 0.0
            with torch.no_grad():
                for _, data in enumerate(test_dataloader):
                    x, t = data
                    mse = model.compute_mse_loss(t.to(device), x.to(device))
                    test_loss += mse
                    cnt = cnt + 1
                test_loss /= cnt
            writer.add_scalar('test_mse_loss', test_loss, epoch)
            
        epoch = epoch + 1
