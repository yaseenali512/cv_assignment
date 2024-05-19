from PIL import Image

import numpy as np
from scipy import signal, fftpack
import cv2 as cv
from skimage import exposure


def apply_low_pass_filter(image, sigma):
    kernel_size = 2 * int(4 * sigma) + 1
    gaussian_kernel = cv.getGaussianKernel(kernel_size, sigma)
    gaussian_kernel_2d = np.outer(gaussian_kernel, gaussian_kernel.T)
    filtered_image = signal.convolve2d(image, gaussian_kernel_2d, mode='valid')
    filtered_image = np.clip(filtered_image, 0, 255)
    filtered_image = np.uint8(filtered_image)
    return filtered_image

def apply_low_pass_butterworth_filter(image, d0, n):
    image_fft = fftpack.fftshift(fftpack.fft2(image))
    rows, cols = image.shape
    u = np.fft.fftfreq(rows, 1)
    v = np.fft.fftfreq(cols, 1)
    V, U = np.meshgrid(v, u)
    D = np.sqrt((U - rows / 2)**2 + (V - cols / 2)**2)
    H = 1 / (1 + (D / d0)**n)
    image_fft_filtered = image_fft * H
    image_filtered = np.abs(fftpack.ifft2(fftpack.ifftshift(image_fft_filtered)))
    image_filtered = np.clip(image_filtered, 0, 255).astype(np.uint8)
    return image_filtered

def apply_high_pass_laplacian_filter(image):
    laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    filtered_image = signal.convolve2d(image, laplacian_kernel, mode='same')
    filtered_image = np.clip(filtered_image, 0, 255)
    filtered_image = np.uint8(filtered_image)
    return filtered_image

def histogram_matching(source_image, reference_image):
    old_shape = source_image.shape
    source = source_image.ravel()
    reference = reference_image.ravel()
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    r_values, r_counts = np.unique(reference, return_counts=True)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    r_quantiles = np.cumsum(r_counts).astype(np.float64)
    r_quantiles /= r_quantiles[-1]
    interp_r_values = np.interp(s_quantiles, r_quantiles, r_values)
    matched = interp_r_values[bin_idx].reshape(old_shape)
    matched = np.clip(matched, 0, 255)
    matched = np.uint8(matched)
    return matched

