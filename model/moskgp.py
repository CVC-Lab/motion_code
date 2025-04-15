import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np
from .duo_hgp import SparseGP


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
        dist_sq = x1_norm + x2_norm.transpose(-1, -2) - 2 * torch.matmul(x1, x2.transpose(-1, -2))  # Shape: (B, N, M)


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

class RationalQuadraticKernel(nn.Module):
    def __init__(self, lengthscale=0.5, variance=1.0, alpha=1.0):
        super(RationalQuadraticKernel, self).__init__()
        self.lengthscale = nn.Parameter(torch.tensor(lengthscale, dtype=torch.float32))
        self.variance = nn.Parameter(torch.tensor(variance, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))

    def forward(self, x1, x2):
        """
        Rational Quadratic kernel between x1 and x2.
        Inputs:
            x1: Tensor (B, N, d)
            x2: Tensor (B, M, d)
        Output:
            Kernel matrix (B, N, M)
        """
        if len(x1.shape) == 2:
            x1 = x1.unsqueeze(0)
        if len(x2.shape) == 2:
            x2 = x2.unsqueeze(0)

        x1_norm = torch.sum(x1 ** 2, dim=-1, keepdim=True)
        x2_norm = torch.sum(x2 ** 2, dim=-1, keepdim=True)

        dist_sq = x1_norm + x2_norm.transpose(-1, -2) - 2 * torch.matmul(x1, x2.transpose(-1, -2))

        # RQ kernel computation
        denom = 2 * self.alpha * self.lengthscale ** 2
        base = 1 + dist_sq / denom
        return torch.clamp(self.variance, 1e-6) * base.pow(-self.alpha)

    def diag(self, x):
        """
        Returns the diagonal of the kernel matrix (B, N)
        """
        return torch.clamp(self.variance, 1e-6) * torch.ones((x.shape[0], x.shape[1]), device=x.device)


class SumOfKernel(nn.Module):
    def __init__(self, L=2, kernel_func=RationalQuadraticKernel):
        super(SumOfKernel, self).__init__()
        if not isinstance(kernel_func, list):
            self.kernels = nn.ModuleList([
                kernel_func(
                    lengthscale=0.5 + 0.1 * i,
                    variance=1.0,
                    alpha=1.0 + 0.1 * i
                ) for i in range(L)
            ])
        self.raw_weights = nn.Parameter(torch.zeros(L))

    def forward(self, x1, x2):
        weights = torch.softmax(self.raw_weights, dim=0)  # ensure positivity
        k_sum = 0
        for w, k in zip(weights, self.kernels):
            k_sum += w * k(x1, x2)
        return k_sum

    def diag(self, x):
        weights = torch.softmax(self.raw_weights, dim=0)
        d_sum = 0
        for w, k in zip(weights, self.kernels):
            d_sum += w * k.diag(x)
        return d_sum

class KernelDynamics(torch.autograd.Function):

    @staticmethod
    def forward(*args, **kwargs):
        return super().forward(*args, **kwargs)
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        return super().setup_context(ctx, inputs, output)
    
    @staticmethod
    def backward(ctx, *grad_outputs):
         # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_outputs.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_outputs.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_outputs.sum(0)

        return grad_input, grad_weight, grad_bias
        return super().backward(ctx, *grad_outputs)


# Sparse Gaussian Process Layer
class SparseGPLayer(nn.Module):
    def __init__(self, input_dim, num_inducing_points, num_latents, kernel_func = RBFKernel, sigma_y=0.1, output_dim=1):
        super(SparseGPLayer, self).__init__()
        self.input_dim = input_dim
        self.num_inducing_points = num_inducing_points
        self.sigma_y = sigma_y
        self.output_dim = output_dim

        # Initialize inducing points randomly
        #self.inducing_point_encoder = nn.Sequential(
        #    nn.Linear(num_latents, num_inducing_points, bias=False).double(),
        #    nn.Sigmoid()
        #              )
        # torch.nn.init.xavier_uniform_(self.inducing_point_encoder[0].weight)
        #self.z = nn.Parameter(torch.ones(num_latents, dtype=torch.float64))
        self.S_m = nn.Parameter(torch.logit(torch.linspace(0.1, 0.9, num_inducing_points, dtype=torch.float64)))
        

        # Variational distribution parameters (mean and covariance)
        self.variational_mean = torch.zeros((num_inducing_points, self.output_dim)).double()
        self.variational_covar = torch.eye(num_inducing_points).double()
        self.precision_covar =  torch.eye(num_inducing_points).double()

        # Kernel function
        self.kernel = kernel_func()
        with torch.no_grad():
            self._initialize_variational_distribution()
    
    def get_inducing_points(self):
        #S_m = self.inducing_point_encoder(self.z).unsqueeze(-1)
        S_m = torch.nn.functional.sigmoid(self.S_m).unsqueeze(-1)
        return S_m

    def _initialize_variational_distribution(self):
        self.inducing_points = self.get_inducing_points().clone()
        #print(self.inducing_points.T)
        K_uu = self.kernel(self.inducing_points, self.inducing_points)  # Covariance of inducing points
        K_uu.squeeze_(0)
        self.variational_covar = K_uu
        L = torch.linalg.cholesky(K_uu + 1e-6 * torch.eye(self.num_inducing_points))
        self.precision_covar = torch.cholesky_inverse(L)
    
    def _update_variational_means_and_covars(self, y_k, K_uf, K_uu, sigma_y, epoch=0):
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
        if False:
            mu_prev =  torch.zeros((self.num_inducing_points, self.output_dim)).double().detach().to(y_k.device)
            precision_prev =  torch.eye(self.num_inducing_points).double().detach().to(y_k.device)
        else:
            mu_prev = self.variational_mean.detach().to(y_k.device)
            precision_prev = self.precision_covar.detach().to(y_k.device)
        B = y_k.shape[0]
        
        jitter_uu = 1e-12 * torch.eye(self.num_inducing_points, device=y_k.device)  # Shape: (1, M, M)
        K_uu_jittered = K_uu + jitter_uu  # Shape: (1, M, M)
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
        print(self.variational_covar.norm())
    

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
        K_uu_jittered = K_uu + 1e-12 * torch.eye(self.num_inducing_points).to(x.device)
        # L_uu = torch.linalg.cholesky(K_uu + 1e-6 * torch.eye(self.num_inducing_points))  # Add small jitter for stability
        A = torch.linalg.solve(K_uu_jittered, K_uf)
        AT = A.permute(0, 2 ,1)
        predictive_mean = torch.einsum("abc,cd->abd", AT, self.variational_mean)
        predictive_covar = K_ff -  torch.bmm(AT, K_uf) + torch.bmm(torch.einsum("abc,cd->abd", AT, self.variational_covar), A)

        return predictive_mean, predictive_covar, K_uu, K_uf, K_ff, self.inducing_points

    def compute_vfe_loss(self, x, y, epoch = 0):
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
        # Variational distribution update
        Sigma_prev = self.variational_covar.detach().to(x.device)
        if self.training:
            self._update_variational_means_and_covars(y, K_uf, K_uu, self.sigma_y, epoch)
        
        jitter_uu = 1e-12 * torch.eye(M, device=x.device)  # Shape: (1, M, M)
        K_uu_jittered = K_uu + jitter_uu  # Shape: (1, M, M)
        # Cholesky decomposition of K_mm
        Sigma = self.variational_covar.to(x.device)
        #print((K_uu @ Q @ K_uu).norm())
        #L, info = torch.linalg.cholesky_ex(K_uu @ Q @ K_uu + 1e-12 * torch.eye(M, device=x.device))  # m * m
        # print(info)
        L = torch.linalg.cholesky(K_uu_jittered)

        # Compute L^{-1}L_Q
        LSigma = torch.linalg.cholesky(Sigma + jitter_uu)
        Linv_LSigma = torch.linalg.solve_triangular(L, LSigma, upper=False)
        
        # Solve triangular system: A = L^{-1} K_mn / sigma_y
        A = torch.linalg.solve_triangular(L, K_uf , upper=False) / self.sigma_y  # B * m * N

        A_tilde = torch.einsum('bmn,mp->bpn', A, Linv_LSigma)

        # This is H_k^T Sigma_k H_k
        #AAT_tilde = torch.einsum('bmn,bpn->bmp', A_tilde, A_tilde)  # B * m * m
        # Compute A @ A.T for each batch
        AAT = torch.einsum('bmn,bpn->bmp', A, A) 

        # drifted y term
        # LK, pivots, info = torch.linalg.ldl_factor_ex(K_uu)

        dx =  torch.linalg.solve(K_uu_jittered, self.variational_mean.to(x.device))
        tilde_y = y - torch.einsum("bnm, np -> bmp", K_uf, dx)
 
        # Solve triangular system: c = LB^{-1} (A @ y) / sigma_y
        #c = torch.linalg.solve_triangular(LB, torch.bmm(A_tilde, tilde_y), upper=False) / self.sigma_y  # B * m
        # c = sqrt(\Sigma_k) @ K_{SS}^{-1} @ K_{ST} @ y
        c = torch.bmm(A_tilde, tilde_y) / self.sigma_y
        # Compute the log marginal likelihood for each batch
        vfe_loss = 0 
        vfe_loss = - N / 2 * torch.log(2 * torch.tensor(torch.pi, device=x.device))  # Scalar
        logdet = - torch.linalg.slogdet(LSigma).logabsdet + 0.5 * torch.linalg.slogdet(Sigma_prev).logabsdet
        vfe_loss += torch.mean(logdet)  # Sum over m for each batch
        vfe_loss -= N / 2 * torch.log(torch.tensor(self.sigma_y).to(x.device) ** 2)  # Scalar
        quadratic = 0
        quadratic += 0.5 / (self.sigma_y ** 2) *  torch.mean(torch.sum(tilde_y ** 2, dim=-1)) # B
        quadratic -= 0.5 * torch.mean(torch.sum(c ** 2, dim=-1))   # B
        vfe_loss += quadratic
        trace = 0
        trace += 0.5 / (self.sigma_y ** 2) * torch.mean(torch.sum(self.kernel.diag(x), dim=-1))  # B
        trace -= 0.5 * torch.mean(torch.vmap(torch.trace)(AAT)) # Trace of AAT for each batch
        vfe_loss += trace
        
        return vfe_loss, logdet, quadratic, trace

    def compute_mse_loss(self):
        #return 0.1 * torch.sum(self.z ** 2)
        return 0
    
    def compute_reg_loss(self):
        return 0

class MultiOutputSparseGPLayer(nn.Module):
    def __init__(self, input_dim, num_inducing_points, num_latents, num_outputs, kernel_func = RBFKernel, sigma_y = 0.1):
        """
        Args:
            input_dim: Dimension of input data (d).
            num_inducing_points: Number of inducing points (M).
            num_latents: Dimension of inducing latents (dz)
            num_outputs: Number of outputs/labels (L).
        """
        super(MultiOutputSparseGPLayer, self).__init__()
        self.input_dim = input_dim
        self.num_inducing_points = num_inducing_points
        self.num_outputs = num_outputs
        self.sigma_y = sigma_y

        # kernel modules
        self.gaussian_processes = nn.ModuleList([SparseGP(input_dim, num_inducing_points, num_latents, kernel_func=kernel_func, sigma_y=sigma_y)  for _ in range(num_outputs)])


    def forward(self, x):
        """
        Forward pass for all GPs.
        Args:
            x: List of input tensors, each of shape (B, N_l, d).
        Returns:
            predictive_means_list: List of predictive means, each of shape (B, L, N_l).
            predictive_covars_list: List of predictive covariances, each of shape (B, L, N_l, N_l).
        """
        predictive_means_list = []
        predictive_covars_list = []
        K_uu_list = []
        K_uf_list = []
        K_ff_list = []
        S_m_list = []

        for l in range(self.num_outputs):
            
            predictive_mean, predictive_covar, K_uu, K_uf, K_ff, S_m = self.gaussian_processes[l](x)
            predictive_means_list.append(predictive_mean)
            predictive_covars_list.append(predictive_covar)
            # in case one needs later on
            K_uu_list.append(K_uu)
            K_uf_list.append(K_uf)
            K_ff_list.append(K_ff)
            S_m_list.append(S_m)


        return predictive_means_list, predictive_covars_list, S_m_list

    def compute_loss(self, x_list, y_list, epoch=0):
        loss = 0
        for l in range(self.num_outputs):
            vfe, logdet, quad, trace = self.gaussian_processes[l].compute_vfe_loss(x_list[l], y_list[l])
            loss = loss + vfe + self.gaussian_processes[l].compute_mse_loss()
        return loss / self.num_outputs


    def predict(self, x, y):
        y_list = y.unsqueeze(0).repeat(self.num_outputs,1,1)
        predictive_mean_list, predictive_cor_list, inducing_points = self.forward(x.unsqueeze(-1))
        for l in range(self.num_outputs):
            std = torch.sqrt(torch.diag(predictive_cor_list[l][0])).reshape(-1)
            #print(self.gaussian_processes[l].S_m.T)
    
        y_predict = torch.stack(predictive_mean_list, dim = 0).squeeze(-1)
        #print(torch.mean((y_predict - y_list ) ** 2, dim=-1))
        return torch.argmin(torch.mean((y_predict - y_list) ** 2, dim=-1), dim=0)


    def forecast(self, x):
        means, covars, S_ms = self.forward(x.unsqueeze(-1)) 
        return means, covars, S_ms


# Define the ELBO Loss
def elbo_loss(y_pred_mean, y_pred_covar, y_true, variational_dist, K_uu):
    """
    Compute the Evidence Lower Bound (ELBO) loss.
    """
    # Log likelihood term
    likelihood_dist = dist.MultivariateNormal(y_pred_mean, covariance_matrix=y_pred_covar + 1e-6 * torch.eye(len(y_true)))
    log_likelihood = likelihood_dist.log_prob(y_true)

    # KL divergence term (between variational distribution and prior)
    prior_dist = dist.MultivariateNormal(torch.zeros_like(variational_dist.mean), covariance_matrix=K_uu)
    kl_divergence = dist.kl.kl_divergence(variational_dist, prior_dist)

    # ELBO = log_likelihood - KL divergence
    return -log_likelihood + kl_divergence

