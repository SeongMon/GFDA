import torch
import numpy as np
from scipy.stats import multivariate_normal

def low_freq_mutate_gau( amp_src, amp_trg, cova1=0, cova2=0):
    mean = [0, 0]
    covariance_matrix = [[cova1, 0], [0, cova2]]
    rv = multivariate_normal(mean, covariance_matrix)
    range_value = 1
    x, y = np.meshgrid(np.linspace(-range_value, range_value, 352), np.linspace(-range_value, range_value, 352))
    pos = np.dstack((x, y))
    pdf_values = rv.pdf(pos)
    amp_src = amp_src.cpu().numpy()
    amp_trg = amp_trg.cpu().numpy()
    pdf_values_normalized = (pdf_values - pdf_values.min()) / (pdf_values.max() - pdf_values.min())
    amp_src = amp_src * (1-pdf_values_normalized) + amp_trg * pdf_values_normalized
    return torch.from_numpy(amp_src).cuda()

@torch.no_grad()
def create_fda_gau(fr, to, cova1, cova2):
    # source : to, target : fr
    fft_src = torch.fft.fft2(to.clone()) 
    fft_trg = torch.fft.fft2(fr.clone())
    amp_src, pha_src = torch.fft.fftshift(torch.abs(fft_src)), torch.angle(fft_src)
    amp_trg, pha_trg = torch.fft.fftshift(torch.abs(fft_trg)), torch.angle(fft_trg)
    amp_src_ = low_freq_mutate_gau( amp_src.clone(), amp_trg.clone(), cova1=cova1, cova2=cova2)
    fft_src_ = torch.fft.ifftshift(amp_src_) * torch.exp( 1j * pha_src )
    src_in_trg = torch.fft.ifft2( fft_src_, dim=(-2, -1))
    src_in_trg = torch.real(src_in_trg)
    return src_in_trg