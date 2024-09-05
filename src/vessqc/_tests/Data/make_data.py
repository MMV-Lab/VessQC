import numpy as np
from bioio.writers import OmeTiffWriter

img = np.zeros([32, 32, 32], dtype=np.float32)
img[:, :, 0] = 255.0
img[:, :, 31] = 255.0
img[:, 0, :] = 255.0
img[:, 31, :] = 255.0
img[0, :, :] = 255.0
img[31, :, :] = 255.0

filename = 'Image.tif'
OmeTiffWriter.save(img, filename, dim_order='ZYX')

pred = np.zeros([32, 32, 32], dtype=np.uint8)
filename = 'Prediction.tif'
OmeTiffWriter.save(pred, filename, dim_order='ZYX')

unc = np.zeros([32, 32, 32], dtype=np.float32)
filename = 'Uncertainty.tif'
OmeTiffWriter.save(unc, filename, dim_order='ZYX')
