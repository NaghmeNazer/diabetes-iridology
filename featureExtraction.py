from numpy import squeeze, real, mean, pi, float16, array, float16, reshape, float32
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel
from skimage.feature import hog
from skimage import feature
import cv2
import numpy as np
from skimage.transform import rescale, resize, downscale_local_mean
import pywt


def hogFeature(normalizedIrisPatch, regions):
    # regions: [(x1, x2), (x3, x4), (x5, x6), ...]
    upperCutHeight = 10
    # HOG Features
    hogFea = []
    for reg in regions:
        croppedImage = normalizedIrisPatch[upperCutHeight:, reg[0]:reg[1]]
        hog_cur = hog(croppedImage, orientations=6, pixels_per_cell=(32, 32), cells_per_block=(1, 1))
        hog_cur = array(hog_cur, float32)
        hogFea.append(hog_cur)

    hogFea = array(hogFea, dtype=float32)
    hogFea = reshape(hogFea, (hogFea.shape[0] * hogFea.shape[1],1))
    hogFea = hogFea.tolist()

    return hogFea

def lbpFeature(normalizedIrisPatch, regions):

    # regions: [(x1, x2), (x3, x4), (x5, x6), ...]
    P = 16
    upperCutHeight = 10
    # LBP Features
    lbpFea = []
    for reg in regions:
        croppedImage = normalizedIrisPatch[upperCutHeight:, reg[0]:reg[1]]
        lbp = feature.local_binary_pattern(croppedImage, 16, 2, method='uniform')
        hist, _ = np.histogram(lbp, normed=True, bins=P + 2, range=(0, P + 2))
        lbpFea.append(hist)

    lbpFea = array(lbpFea, dtype=float32)
    lbpFea = reshape(lbpFea, (lbpFea.shape[0] * lbpFea.shape[1],1))
    lbpFea = lbpFea.tolist()

    return lbpFea

def gaborFeature(normalizedIrisPatch, regions):
    # regions: [(x1, x2), (x3, x4), (x5, x6), ...]
    upperCutHeight = 10
    # Gabor Features
    kernels = []
    freqs = [0.1, 0.2, 0.3, 0.4, 0.5]
    nTheta = 8
    for theta in range(nTheta):
        theta = theta / float16(nTheta) * pi
        sigma = 1
        for frequency in freqs:
            kernel = real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)

    gaborFea = []

    for reg in regions:
        croppedImage = normalizedIrisPatch[upperCutHeight:, reg[0]:reg[1]]
        gaborFea_cur = []
        for k, kernel in enumerate(kernels):
            filteredIris = ndi.convolve(croppedImage, kernel, mode='wrap')
            gaborFea_cur.append(mean(filteredIris * filteredIris))
        gaborFea_cur = array(gaborFea_cur, float32)
        gaborFea.append(gaborFea_cur)

    gaborFea = array(gaborFea, dtype=float32)
    gaborFea = reshape(gaborFea, (gaborFea.shape[0] * gaborFea.shape[1],1))
    gaborFea =gaborFea.tolist()

    return gaborFea


def extract_image_feature(image, regions, downSampleSize):
    # regions: [(x1, x2), (x3, x4), (x5, x6), ...]
    upperCutHeight = 10
    # Pixel Features
    pixelFea = []
    for reg in regions:
        croppedImage    = image[upperCutHeight:, reg[0]:reg[1]]
        downSampledReg  = rescale(croppedImage, 1.0 / float16(downSampleSize), preserve_range=True)
        pixelFea.append(reshape(downSampledReg, (downSampledReg.shape[0]*downSampledReg.shape[1],)))

    pixelFea = array(pixelFea, dtype=float32)
    pixelFea = reshape(pixelFea, (pixelFea.shape[0]*pixelFea.shape[1], 1))
    pixelFea = pixelFea.tolist()

    return pixelFea
