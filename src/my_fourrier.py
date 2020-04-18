import numpy as np
import math
from helpers import pad_image


from scipy.sparse import csr_matrix
import scipy.sparse


def DFT(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    
    temp = np.e ** ((-2j * math.pi * k * n) / N)
    
    return np.dot(x, temp)
    
def IDFT(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    
    temp = np.e ** ((2j * math.pi * k * n) / N)
    
    return np.dot(x, temp) / N


def FFT(x, size = 256):
    if len(x) > size:
        even = FFT(x[::2])
        odd = FFT(x[1::2])
        
        N = len(x)
        n = np.arange(N)
        
        temp = np.e ** ((-2j * math.pi * n) / N)
        return np.concatenate((even + temp[:int(math.floor(N/2))] * odd,
                              even + temp[int(math.ceil(N/2)):] * odd), axis=0)
    
    else:
        return DFT(x)

def IFFT(x):
    temp = np.flip(x[1:])
    x = np.concatenate(([x[0]], temp), axis = 0)
    
    return FFT(x).real / len(x)


def FFT2(x):
    result = np.empty_like(x, dtype = complex)
    w, h = x.shape
    
    for i in range(w):
        result[i, :] = FFT(x[i, :])
        
    for i in range(h):
        result[:, i] = FFT(result[:, i])
        
    return result

def DFT2(x):
    result = np.empty_like(x, dtype = complex)
    w, h = x.shape
    
    for i in range(w):
        result[i, :] = DFT(x[i, :])
        
    for i in range(h):
        result[:, i] = DFT(result[:, i])
        
    return result

def IFFT2_helper(x):
    temp = np.flip(x[1:])
    x = np.concatenate(([x[0]], temp), axis = 0)
    
    return FFT(x)

def IFFT2(x):
    result = np.empty_like(x, dtype = complex)
    w, h = x.shape
    
    for i in range(w):
        result[i, :] = IFFT2_helper(x[i, :])
        
    for i in range(h):
        result[:, i] = IFFT2_helper(result[:, i])
        
    return result / (w * h)


def denoise_frequency(img, threshold_low = 0.1, threshold_high = 0.1):
    fft_img = img.copy()
    
    h, w = fft_img.shape
    
    fft_img[int(threshold_low * h):-int(threshold_high * h), :] = 0
    fft_img[:, int(threshold_low * w):-int(threshold_high * w)] = 0
    
    non_zero_count = np.count_nonzero(fft_img)
    print("amount of non-zeros: ", non_zero_count)
    print("fraction of non-zero coefficient: ", non_zero_count / fft_img.size)
    
    denoised = IFFT2(fft_img)
    return denoised.real

def denoise_cutoff(img, cutoff_low = 0.1, cutoff_high = 0.9):
    
    fft_img = img.copy()
    
    fft_max = fft_img.max()
    fft_min = fft_img.min()
    
    cutoff_max = ((fft_max - fft_min) * cutoff_high) + fft_min
    cutoff_min = ((fft_max - fft_min) * cutoff_low) + fft_min
    
    fft_img[fft_img < fft_min] = cutoff_min
    fft_img[fft_img > fft_max] = cutoff_max
    
    non_zero_count = np.count_nonzero(fft_img)
    print("amount of non-zeros: ", non_zero_count)
    print("fraction of coefficient: ", non_zero_count / fft_img.size)

    denoised = IFFT2(fft_img)    
    return denoised.real

def compress_frequency_cutoff(img, filename, compression_ratio = 0.6):
    fft_img = img.copy()
    
    c_h = int(math.sqrt(1 - compression_ratio) * (fft_img.shape[0] / 2))
    c_w = int(math.sqrt(1 - compression_ratio) * (fft_img.shape[1] / 2))
    
    fft_img[c_h:-c_h, :] = 0+0.j
    fft_img[:, c_w:-c_w] = 0+0.j
    
    nonzero_pre_compression = np.count_nonzero(img)
    nonzero_post_compression = np.count_nonzero(fft_img)
    
    print("nonzero values: ", np.count_nonzero(fft_img))
    print("compression ratio: ", 1 - (nonzero_post_compression / nonzero_pre_compression))
    
    temp_sparse = csr_matrix(fft_img)
    scipy.sparse.save_npz(filename + '_' + str(compression_ratio) + ".npz", temp_sparse)
    
    return temp_sparse
    
def compress_magnitude_cutoff(img, compression_ratio = 0.6):

    fft_img = img.copy()
    
    compression_size = int((1 - compression_ratio) * fft_img.size)
    
    cutoff_value = np.sort(np.abs(fft_img).flatten())[::-1][compression_size]
    
    fft_img[np.abs(fft_img) < cutoff_value] = 0+0.j
    
    nonzero_pre_compression = np.count_nonzero(img)
    nonzero_post_compression = np.count_nonzero(fft_img)
    
    print("nonzero values: ", np.count_nonzero(fft_img))
    print("compression ratio: ", 1 - (nonzero_post_compression / nonzero_pre_compression))
    
    
    temp_sparse = csr_matrix(fft_img)
    
    return temp_sparse
    
    