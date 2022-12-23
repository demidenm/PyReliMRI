import nibabel as nib
import numpy as np

imgdata = np.random.randn(64,64,32)
img = nib.Nifti1Image(imgdata, np.eye(4))
nib.save(img, '../testdata/test1.nii.gz')
