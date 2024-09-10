import numpy as np
from bioio import BioImage
from bioio.writers import OmeTiffWriter
import bioio_tifffile

# Load the TIFF files
img = BioImage('Prediction.tif', reader=bioio_tifffile.Reader)
print('img.dims', img.dims)
prediction = img.get_image_data("ZYX", T=0, C=0)

img = BioImage('Uncertainty.tif', reader=bioio_tifffile.Reader)
print('img.dims', img.dims)
uncertainty = img.get_image_data("ZYX", T=0, C=0)

# Transfer the prediction values to the 3D matrix of uncertainties
where1 = np.where(prediction==1)
uncertainty[where1] = 1.0

# Calculate values for the uncertainties
x = np.asarray([i*i for i in range(0, 32)])
y = [i*i for i in range(0, 32)]
y.reverse()
y = np.asarray(y)
z = np.zeros([32, 32])
z2 = np.zeros([32, 32])

for i in range(0, 32):
    z[i, :] = x
    z2[:, i] = y

z = np.sqrt(z + z2)
z = z / np.sqrt(2 * 31**2)
z = np.floor(9.0 * z) + 1.0
z /= 10.0

# Transfer the values to the uncertainty array
for i in range(0, 32):
    uncertainty[i, :, :] *= z

# Save the result
OmeTiffWriter.save(uncertainty, 'Uncertainty.tif', dim_order='ZYX')
