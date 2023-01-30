import torch
import numpy as np


def generate_data(batch_size, N, M, snr_low, snr_high, u, Es):
    #x_ = torch.randint(int(u)+1, (batch_size, N,1))+1
    x = np.random.randint(int(u)+1, size=(batch_size, 2*N,1))
    x = 2*(x+1)-int(u)-2
    x_c= x[:, 0:N,:]+1j*x[:, N:2*N,:]
    H_ = np.random.randn(batch_size, 2*M, 2*N)/np.sqrt(2)
    H_c = H_[:, 0:M, 0:N]+1j*H_[:, M:2*M, N:2*N]

    # SNR_ =   (snr_high - snr_low) * np.random.rand(batch_size, 1, 1) + snr_low
    # sigma_snr = np.sqrt( N* (10 ** ( - (SNR_) / 10 )) )

    sigma_low = np.sqrt( N* (10 ** ( - (snr_low) / 10 )) )
    sigma_high = np.sqrt( N* (10 ** ( - (snr_high) / 10 )) )
    sigma_snr =  (sigma_high - sigma_low) * np.random.rand(batch_size, 1, 1) + sigma_low

    sigma2 = (sigma_snr*np.sqrt(float(Es))/np.sqrt(2))**2
    noise = np.multiply( np.random.randn(batch_size, 2*M, 1), np.sqrt(sigma2))
    noise_c = ( noise[:, 0:M, :]+1j*noise[:, M:2*M, :] )
    y_c = np.add(np.matmul(H_c, x_c), noise_c)
    y = np.concatenate((np.real(y_c), np.imag(y_c)), 1)
    H_row1 = np.concatenate((np.real(H_c), -np.imag(H_c)), 2)
    H_row2 = np.concatenate((np.imag(H_c), np.real(H_c)), 2)
    H = np.concatenate((H_row1, H_row2), 1)

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    H = torch.from_numpy(H)
    sigma2 = torch.from_numpy(sigma2)

    return x_c, x, y_c, y, H_c, H, noise_c, noise, sigma2

def BR(x, x_pre):
    return np.sum( np.where(np.abs(x-x_pre)<1e-10, 1, 0) )

def de_QAM(x, cons, batch_size, N):
    cons_expand = cons.repeat(batch_size, x.size()[1],1)
    x_expand = x.repeat(1, 1, cons.size()[0])
    temp = torch.min( torch.abs(x_expand-cons_expand), 2 )
    xout = cons[temp[1].reshape(-1,)].reshape(batch_size, N, -1)
    return xout