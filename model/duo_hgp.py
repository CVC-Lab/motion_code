"""
Define Hamiltonian GP here


"""

import torch
import torch.nn as nn
import torch.optim as optim

# Define the RBF Kernel
class RBFKernel(nn.Module):
    def __init__(self, lengthscale=0.5, variance=1.0):
        super(RBFKernel, self).__init__()
        self.lengthscale = nn.Parameter(torch.tensor(lengthscale, dtype=torch.float32))
        self.variance = nn.Parameter(torch.tensor(variance, dtype=torch.float32))

    def forward(self, x1, x2):
        """
        Compute the RBF kernel matrix for batched inputs.
        Inputs:
            x1: Tensor of shape (B, N, d)
            x2: Tensor of shape (B, M, d)
        Output:
            Kernel matrix of shape (B, N, M)
        """
        if len(x1.shape)==2:
            x1.unsqueeze_(0)
        if len(x2.shape)==2:
            x2.unsqueeze_(0)
        # Compute squared norms of x1 and x2
        x1_norm = torch.sum(x1 ** 2, dim=-1, keepdim=True)  # Shape: (B, N, 1)
        x2_norm = torch.sum(x2 ** 2, dim=-1, keepdim=True)  # Shape: (B, M, 1)
        # Compute pairwise squared distances using broadcasting
        dist_sq = x1_norm + x2_norm.transpose(-1, -2) - 2.0 * torch.matmul(x1, x2.transpose(-1, -2))  # Shape: (B, N, M)


        # Compute the RBF kernel
        return torch.clamp(self.variance, 1e-6) * torch.exp(-0.5 * dist_sq / self.lengthscale ** 2)
    
    def diag(self, x):
        """
        Compute the RBF kernel matrix for batched inputs but only diagonal terms.
        Inputs:
            x: Tensor of shape (B, N, d)
        Output:
            Kernel matrix of shape (B, N)
        """
        return torch.clamp(self.variance, 1e-6) * torch.ones((x.shape[0], x.shape[1])).to(x.device)


class SparseGP(nn.Module):
    """Single-layer Sparse Gaussian Process."""

    def __init__(self, input_dim, num_inducing_points, num_latents, kernel_func = RBFKernel, sigma_y=0.1, output_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.num_inducing_points = num_inducing_points
        self.sigma_y = sigma_y
        self.output_dim = output_dim

        self.inducing_point_encoder = nn.Sequential(
           nn.Linear(num_latents, num_inducing_points, bias=False),
           nn.LeakyReLU()
                     )
        torch.nn.init.xavier_uniform_(self.inducing_point_encoder[0].weight)
        self.z = nn.Parameter(torch.ones(num_latents, dtype=torch.float32))
        #self.S_m = nn.Parameter(torch.linspace(0.1, 0.9, num_inducing_points, dtype=torch.float32))
        

        # Variational distribution parameters (mean and covariance)
        self.variational_mean = torch.zeros((num_inducing_points, self.output_dim))
        self.variational_covar = torch.eye(num_inducing_points)
        self.precision_covar =  torch.eye(num_inducing_points)

        # Kernel function
        self.kernel = kernel_func()
    
    def get_inducing_points(self):
        S_m = self.inducing_point_encoder(self.z).unsqueeze(-1)
        return S_m
    
    def _update_variational_means_and_covars(self, y_k, K_uf, K_uu, sigma_y):
        """
        Update the variational mean and covariance using the given update rule.
        
        Args:
            mu_prev: Previous variational mean, shape (M,).
            Sigma_prev: Previous variational covariance, shape (M, M).
            y_k: Observed data, shape (N,).
            K_Tk_Sm: Kernel matrix between test points and inducing points, shape (N, M).
            K_Sm_Sm: Kernel matrix between inducing points, shape (M, M).
            sigma_y: Noise variance.
            
        Returns:
            mu_k: Updated variational mean, shape (M,).
            Sigma_k: Updated variational covariance, shape (M, M).
        """
        # Compute H_k = K_Tk_Sm @ inv(K_Sm_Sm)
        B = y_k.shape[0]
        
        jitter_uu = 1e-6 * torch.eye(self.num_inducing_points, device=y_k.device)  # Shape: (1, M, M)
        K_uu_jittered = K_uu + jitter_uu  # Shape: (1, M, M)
        if True:
            mu_prev =  torch.zeros((self.num_inducing_points, self.output_dim)).float().detach().to(y_k.device)
            Lu = torch.linalg.cholesky(K_uu_jittered)
            precision_prev = torch.cholesky_inverse(Lu)
            #precision_prev =  torch.zeros(self.num_inducing_points).float().detach().to(y_k.device)
        else:
            mu_prev = self.variational_mean.detach().to(y_k.device)
            precision_prev = self.precision_covar.detach().to(y_k.device)
        
        #K_uu_inv = torch.linalg.pinv(K_uu_jittered)  # m * m
        # K_Tk_Sm = K_Sm_Tk.permute(0, 2, 1)
        # H_k = torch.mean(K_Tk_Sm, dim=0) @ K_uu_inv  # Shape: (N, M)
        # H_k = H_k
        # LK, pivots, info = torch.linalg.ldl_factor_ex(K_uu)
        H_k_T = torch.linalg.solve(K_uu_jittered, K_uf)

        # Compute Sigma_k^{-1} = Sigma_prev^{-1} + H_k^T @ inv(V_k) @ H_k
        sigma_y2_inv = 1./ sigma_y ** 2
        precision_next = precision_prev + sigma_y2_inv * torch.mean(torch.einsum("bmn,bpn->bmp", H_k_T, H_k_T), dim=0) # Shape: (M, M)

        # Compute Sigma_k = inv(Sigma_k^{-1})
        L = torch.linalg.cholesky(precision_next)
        Sigma_k = torch.cholesky_inverse(L)  # Shape: (M, M)
        
        # Compute mu_k = Sigma_k @ {H_k^T @ inv(V_k) @ y_k + Sigma_prev^{-1} @ mu_prev}
        mu_k = torch.linalg.solve(precision_next, sigma_y2_inv *  torch.mean(torch.bmm(H_k_T, y_k), dim=0) + precision_prev @ mu_prev)  # Shape: (M,)
        # if epoch>0:
        #     self.variational_mean = mu_prev.to(y_k.device)
        # else:
        self.variational_mean = mu_k.to(y_k.device)
        self.variational_covar = Sigma_k.to(y_k.device)
        self.precision_covar = precision_next.to(y_k.device)

    def forward(self, x):
        """
        Forward pass for the sparse GP layer.
        """
        # Compute kernel matrices
        self.inducing_points = self.get_inducing_points().clone()

        K_uu = self.kernel(self.inducing_points, self.inducing_points)  # Covariance of inducing points
        K_uu.squeeze_(0)
        K_uf = self.kernel(self.inducing_points, x)  # Cross-covariance between inducing points and inputs
        
        K_ff = self.kernel(x, x)  # Covariance of inputs (not used directly in sparse GP)

        # Predictive mean and covariance
        K_uu_jittered = K_uu + 1e-6 * torch.eye(self.num_inducing_points).to(x.device)
        # L_uu = torch.linalg.cholesky(K_uu + 1e-6 * torch.eye(self.num_inducing_points))  # Add small jitter for stability
        A = torch.linalg.solve(K_uu_jittered, K_uf)
        AT = A.permute(0, 2 ,1)
        predictive_mean = torch.einsum("abc,cd->abd", AT, self.variational_mean)
        predictive_covar = K_ff -  torch.bmm(AT, K_uf) + torch.bmm(torch.einsum("abc,cd->abd", AT, self.variational_covar), A)

        return predictive_mean, predictive_covar, K_uu, K_uf, K_ff, self.inducing_points

    def compute_vfe_loss(self, x, y):
        """
        Compute the loss for one GP
        Args:
            x: Input tensor of shape (B, N_l, d).
            y: Target tensor of shape (B, N_l).
        Returns:
            loss: Scalar loss value.
        """
        
        B, N, d = x.shape
        M = self.num_inducing_points
        self.inducing_points = self.get_inducing_points().clone()
        #print(self.inducing_points.T)
        K_uu = self.kernel(self.inducing_points, self.inducing_points)  # Covariance of inducing points
        K_uu.squeeze_(0)
        K_uf = self.kernel(self.inducing_points, x)  # Cross-covariance between inducing points and inputs

        jitter_uu = 1e-6 * torch.eye(M, device=x.device)  # Shape: (1, M, M)
        K_uu_jittered = K_uu + jitter_uu  # Shape: (1, M, M)
        L = torch.linalg.cholesky(K_uu_jittered)

        with torch.no_grad():
            self._update_variational_means_and_covars(y, K_uf, K_uu, self.sigma_y)
        # Solve triangular system: A = L^{-1} K_mn / sigma_y
        A = torch.linalg.solve_triangular(L, K_uf , upper=False) / self.sigma_y  # B * m * N
        AAT = torch.einsum('bmn,bpn->bmp', A, A) 
        LB = torch.linalg.cholesky(AAT + torch.eye(M, device=x.device))
        # Solve triangular system: c = LB^{-1} (A @ y) / sigma_y
        c = torch.linalg.solve_triangular(LB, torch.bmm(A, y) , upper=False) / self.sigma_y  # B * m
        # Compute the log marginal likelihood for each batch
        vfe_loss = 0 
        vfe_loss = - N / 2 * torch.log(2 * torch.tensor(torch.pi, device=x.device))  # Scalar
        logdet = 0
        logdet  = torch.mean(-torch.linalg.slogdet(LB).logabsdet )
        vfe_loss += logdet  # Sum over m for each batch
        vfe_loss -= N / 2 * torch.log(torch.tensor(self.sigma_y).to(x.device) ** 2)  # Scalar
        quadratic = 0
        quadratic += 0.5 / (self.sigma_y ** 2) *  torch.mean(torch.sum(y ** 2, dim=-1)) # B
        quadratic -= 0.5 * torch.mean(torch.sum(c ** 2, dim=-1))   # B
        vfe_loss += quadratic
        trace = 0
        trace += 0.5 / (self.sigma_y ** 2) * torch.mean(torch.sum(self.kernel.diag(x), dim=-1))  # B
        trace -= 0.5 * torch.mean(torch.vmap(torch.trace)(AAT)) # Trace of AAT for each batch
        vfe_loss += trace
        
        return vfe_loss, logdet, quadratic, trace

    def compute_mse_loss(self, x, y):
        y_pred, _, _, _, _, _ = self.forward(x)
        return torch.nn.MSELoss()(y_pred, y)
    
    def compute_mse_loss(self):
        # y_pred, _, _, _, _, _ = self.forward(x)
        # return torch.nn.MSELoss()(y_pred, y)
        return torch.sum(self.z ** 2)

class Observer(nn.Module):
    pass