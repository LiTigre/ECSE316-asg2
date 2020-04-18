import sys
sys.path.insert(0, './src')
from my_fourrier import *
from helpers import *

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cv2
import numpy as np



def mode_1(img_name):
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    img = pad_image(img)

    result = FFT2(img)

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title("Original image"), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(result.real, norm=colors.LogNorm(vmin=5))
    plt.title("FFT form image"), plt.xticks([]), plt.yticks([])

    plt.show()

def mode_2(img_name):
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    img = pad_image(img)

    result = FFT2(img)

    denoised = denoise_frequency(result)

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title("Original image"), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(denoised.real, cmap='gray')
    plt.title("Denoised image"), plt.xticks([]), plt.yticks([])

    plt.show()


def mode_3(img_name):

    filename = img_name.split('.')[0]

    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    img = pad_image(img)

    fft_img = FFT2(img)

    compressed_1 = compress_frequency_cutoff(fft_img, filename, 0)
    compressed_2 = compress_frequency_cutoff(fft_img, filename, 0.25)
    compressed_3 = compress_frequency_cutoff(fft_img, filename, 0.4)
    compressed_4 = compress_frequency_cutoff(fft_img, filename, 0.6)
    compressed_5 = compress_frequency_cutoff(fft_img, filename, 0.8)
    compressed_6 = compress_frequency_cutoff(fft_img, filename, 0.95)

    uncompressed_1 = IFFT2(compressed_1.toarray())
    uncompressed_2 = IFFT2(compressed_2.toarray())
    uncompressed_3 = IFFT2(compressed_3.toarray())
    uncompressed_4 = IFFT2(compressed_4.toarray())
    uncompressed_5 = IFFT2(compressed_5.toarray())
    uncompressed_6 = IFFT2(compressed_6.toarray())

    plt.subplot(321), plt.imshow(uncompressed_1.real, cmap='gray')
    plt.title("0% compression"), plt.xticks([]), plt.yticks([])

    plt.subplot(322), plt.imshow(uncompressed_2.real, cmap = 'gray')
    plt.title("25% compression"), plt.xticks([]), plt.yticks([])

    plt.subplot(323), plt.imshow(uncompressed_3.real, cmap = 'gray')
    plt.title("40% compression"), plt.xticks([]), plt.yticks([])

    plt.subplot(324), plt.imshow(uncompressed_4.real, cmap = 'gray')
    plt.title("60% compression"), plt.xticks([]), plt.yticks([])

    plt.subplot(325), plt.imshow(uncompressed_5.real, cmap = 'gray')
    plt.title("80% compression"), plt.xticks([]), plt.yticks([])

    plt.subplot(326), plt.imshow(uncompressed_6.real, cmap = 'gray')
    plt.title("95% compression"), plt.xticks([]), plt.yticks([])

    plt.show()


def mode_4():
    p_size = np.loadtxt('problem_sizes')
    dft_data = np.loadtxt('dft_times')
    fft_data = np.loadtxt('fft_times')

    dft_mean = [np.mean(x) for x in dft_data]
    dft_std = [np.std(x) for x in dft_data]

    fft_mean = [np.mean(x) for x in fft_data]
    fft_std = [np.std(x) for x in fft_data]

    print('problem sizes: ', p_size)
    print('dft mean times: ', dft_mean)
    print('dft std times: ', dft_std)

    print('fft mean times: ', fft_mean)
    print('fft std times: ', fft_std)

    plt.errorbar(p_size, dft_mean, yerr=dft_std, fmt='-o', label = 'dft')
    plt.errorbar(p_size, fft_mean, yerr=fft_std, fmt='-o', label = 'fft')

    plt.xlabel('exponent size of matrix (2^x)')
    plt.ylabel('runtime (seconds)')
    plt.title('runtime graph of fourrier transforms with various problem sizes')
    plt.legend()

    plt.show()


if __name__ == '__main__':

    mode = 1
    filename = 'moonlanding.png'

    if len(sys.argv) > 1:
        mode = int(sys.argv[1])

        if len(sys.argv) > 2:
            filename = sys.argv[2]

    # print(mode)
    # print(filename)

    if mode == 1:
        mode_1(filename)

    elif mode == 2:
        mode_2(filename)

    elif mode == 3:
        mode_3(filename)
        
    elif mode == 4:
        mode_4()

