import torch
import numpy as np
from .moskgp import MultiOutputSparseGPLayer

class MultiPhaseAMModel(torch.nn.Module):
    def __init__(self, input_dim, num_inducing_points, num_latents, num_outputs, sigma_y = 0.1):
        """
        input_size: original signal length (must be power of 2)
        """
        super(MultiPhaseAMModel, self).__init__()
        self.input_dim = input_dim
        self.num_inducing_points = num_inducing_points
        self.num_outputs = num_outputs
        self.sigma_y = sigma_y
        
        # inner model    
        self.freq_model = MultiOutputSparseGPLayer(input_dim, num_inducing_points, num_latents, num_outputs, sigma_y=0.1)
        self.wavelet_model = MultiOutputSparseGPLayer(input_dim, num_inducing_points, num_latents, num_outputs, sigma_y=0.1, kernel_func=SumOfKernel)

    def _fft_feat(self, x: torch.Tensor, nt=100):
        fft_feat = torch.fft.fft(x)

        return fft_feat.abs()[:,:nt//2].to(x.device)
    
    def _fft_freq(self, x: torch.Tensor, nt=100):
        B = x.shape[0]
        t = torch.linspace(0, 1, nt+1)[:-1]
        fft_freq = torch.fft.fftfreq(nt, 1./nt)
        return t[:nt//2].view(1, -1).repeat(B,1).to(x.device)
    
        
    def _haar_transform(self, x: torch.Tensor):
        """
        Perform Haar transform on batched input.
        
        Input:
            x: Tensor of shape (B, N) — N must be power of 2
        Output:
            List of tensors: [approx (B, 1), detail_L (B, 2^{L-1}), ..., detail_1 (B, N/2)]
        """
        B, N = x.shape
        assert (N & (N - 1)) == 0, "Input length must be a power of 2"

        coeffs = []
        s = x.clone()

        while s.shape[1] > 1:
            avg = (s[:, 0::2] + s[:, 1::2]) / np.sqrt(2)
            diff = (s[:, 0::2] - s[:, 1::2]) / np.sqrt(2)
            coeffs.append(diff)
            s = avg

        coeffs.append(s)  # Final approximation of shape (B, 1)
        return torch.cat(coeffs[::-1], dim = 1).to(x.device)  # Return: [approx, detail_L, ..., detail_1]
    
    def _haar_inverse_transform(self, coeffs: torch.Tensor):
        """
        coeffs: tensor of shape (B, N), where N = 2^L
                assumed ordered as [approx, detail_L, ..., detail_1]
        Returns: (B, N) reconstructed signals
        """
        B, N = coeffs.shape
        assert (N & (N - 1)) == 0, "Length must be power of 2"

        levels = int(np.log2(N))
        ptr = 1
        s = coeffs[:, 0:1]

        for i in range(levels):
            d_len = 2**i
            detail = coeffs[:, ptr:ptr + d_len]
            ptr += d_len

            up = torch.zeros((B, 2 * d_len), device=coeffs.device, dtype=coeffs.dtype)
            up[:, 0::2] = (s + detail) / np.sqrt(2)
            up[:, 1::2] = (s - detail) / np.sqrt(2)
            s = up

        return s

    @torch.no_grad()
    def transmute(self, x: torch.Tensor):
        if len(x.shape) == 3:           
            return self._fft_feat(x.squeeze()), self._haar_transform(x.squeeze())
        elif len(x.shape) == 2:
            return self._fft_feat(x), self._haar_transform(x)
        else:
            raise NotImplemented

    def forward(self, x):
        """
        x: Tensor of shape (B, N, d), where N is input_size
        """
        # Pass through model
        if len(x.shape)==2:
            x.unsqueeze_(-1)
        output_freq = self.freq_model(self._fft_freq(x).unsqueeze(-1))
        output_wavelet = self.wavelet_model(x)
        return output_freq, output_wavelet

    
    def predict(self, x, y):
        fft_y = self._fft_feat(y)
        fft_x = self._fft_freq(y)
        y_list = fft_y.unsqueeze(0).repeat(self.num_outputs,1,1)
        fft_output, _ = self.forward(fft_x.unsqueeze(-1))
        predictive_mean_list, predictive_cor_list, _ = fft_output
        for l in range(self.num_outputs):
            std = torch.sqrt(torch.diag(predictive_cor_list[l][0])).reshape(-1)
            #print(self.gaussian_processes[l].S_m.T)
    
        y_predict = torch.stack(predictive_mean_list, dim = 0).squeeze(-1)
        #print(torch.mean((y_predict - y_list ) ** 2, dim=-1))
        return torch.argmin(torch.mean((y_predict - y_list) ** 2, dim=-1), dim=0)


    def forecast(self, x):
        output_freq, _ = self.forward(x) 
        means, covars, S_ms = output_freq
        return means, covars, S_ms
        

    def compute_loss(self, x_list, y_list, epoch=0):
        # Transmute y_list 
        x_freq_list = []
        y_freq_list = []
        y_wavelet_list = []
        
        for l in range(self.num_outputs):
            fft_y, wavelet_y = self.transmute(y_list[l])
            x_freq_list.append(self._fft_freq(x_list[l]).unsqueeze(-1))
            y_freq_list.append(fft_y.unsqueeze(-1))
            y_wavelet_list.append(wavelet_y.unsqueeze(-1))
        loss = self.freq_model.compute_loss(x_freq_list, y_freq_list, epoch=epoch) + self.wavelet_model.compute_loss(x_list, y_wavelet_list, epoch=epoch)
        return loss
