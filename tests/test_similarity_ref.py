import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from imgreliability.similarity import (
    image_similarity,
    permute_images
)
from imgreliability.icc import (
    sumsq_icc,
    aov_icc,
    peng_icc
)
from collections import namedtuple
from nibabel import Nifti1Image
import nibabel as nib
from scipy.stats import multivariate_normal as multivar_norm


def generate_img_pair(r: float, dir: Path, use_mask: bool = False, tol: float = .001,
                      imgdims: list = None,
                      maskdims: list = None) -> namedtuple:
    """
    r: correlation bw images
    dir: Path for saving files
    use_mask: bool, create mask and mask data
    tol: tolerance for correlation value - lower than .001 will make it really slow
    returns:
        images: path to two image files with specified correlation
        mask: path to mask image, all ones if use_mask==False
    """
    imgpair = namedtuple("ImgPair", "tol r images maskimg")
    imgpair.images = []
    imgpair.r = r
    imgpair.tol = tol

    rng = np.random.default_rng()

    if imgdims is None:
        imgdims = [64, 64, 32]
    nvox = np.prod(imgdims)
    if use_mask:
        if maskdims is None:
            maskdims = [round(i/2) for i in imgdims]
        mask = np.zeros(imgdims).astype('int')
        mask[:maskdims[0], :maskdims[1], :maskdims[2]] = 1
    else:
        mask = np.ones(imgdims)
        
    maskvox = mask.reshape(nvox)

    empirical_r = 10
    while (np.abs(empirical_r - r) > tol):
        data = rng.multivariate_normal(mean=[0,0],
            cov=[[1,r],[r,1]], size=nvox)
        empirical_r = np.corrcoef(data[maskvox == 1, :].T)[0,1]
    
    for i in range(2):
        imgpair.images.append(dir / f'testimg_{i}.nii.gz')
        tmpimg = Nifti1Image((maskvox * data[:, i]).reshape(imgdims),
            affine=np.eye(4))
        tmpimg.to_filename(imgpair.images[-1])
    imgpair.mask = dir / 'mask.nii.gz'
    maskimg = Nifti1Image(mask, affine=np.eye(4))
    maskimg.to_filename(imgpair.mask)
    return imgpair
    

@pytest.fixture(scope="session")
def image_pair(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("data")
    return generate_img_pair(r=0.5, dir=tmpdir)


def test_image_pair_smoke(image_pair):
    assert image_pair.images is not None

def test_image_pair_images(image_pair):
    for imgfile in image_pair.images + [image_pair.mask]:
        img = nib.load(imgfile)
        assert img is not None

@pytest.mark.parametrize("measure", ['Dice', 'Jaccard'])
def test_image_similarity(image_pair, measure):
    imgsim = image_similarity(
        image_pair.images[0], image_pair.images[1], image_pair.mask,
        thresh=1, similarity_type=measure
    )
    assert imgsim is not None

#@pytest.mark.parametrize("corr", [.50, .60,.45])
def test_calculate_icc1():
    import seaborn as sns
    data = sns.load_dataset('anagrams')
    # subset to only divided attnr measure occ
    a_wd = data[data['attnr'] == 'divided']
    a_ld = pd.DataFrame(
        pd.melt(a_wd,
                id_vars="subidr",
                value_vars=["num1", "num2", "num3"],
                var_name="sess",
                value_name="vals"))

    icc = peng_icc(df_long=a_ld, sub_var="subidr",
                   sess_var="sess", values="vals", icc_type='icc_1')

    assert np.allclose(round(icc, 2), -0.05)


def test_calculate_icc2():
    import seaborn as sns
    data = sns.load_dataset('anagrams')
    # subset to only divided attnr measure occ
    a_wd = data[data['attnr'] == 'divided']
    a_ld = pd.DataFrame(
        pd.melt(a_wd,
                id_vars="subidr",
                value_vars=["num1", "num2", "num3"],
                var_name="sess",
                value_name="vals"))

    icc = peng_icc(df_long=a_ld, sub_var="subidr",
                   sess_var="sess",values="vals", icc_type='icc_2')
    assert np.allclose(round(icc, 2), 0.11)


def test_calculate_icc3():
    import seaborn as sns
    data = sns.load_dataset('anagrams')
    # subset to only divided attnr measure occ
    a_wd = data[data['attnr'] == 'divided']
    a_ld = pd.DataFrame(
        pd.melt(a_wd,
                id_vars="subidr",
                value_vars=["num1", "num2", "num3"],
                var_name="sess",
                value_name="vals"))

    icc = aov_icc(df_long=a_ld, sub_var="subidr",
                   sess_var="sess", values="vals", icc_type='icc_3')
    assert np.allclose(round(icc, 2), 0.20)


def test_calculate_icc3():
    import seaborn as sns
    data = sns.load_dataset('anagrams')
    # subset to only divided attnr measure occ
    a_wd = data[data['attnr'] == 'divided']
    a_ld = pd.DataFrame(
        pd.melt(a_wd,
                id_vars="subidr",
                value_vars=["num1", "num2", "num3"],
                var_name="sess",
                value_name="vals"))

    icc = aov_icc(df_long=a_ld, sub_var="subidr",
                   sess_var="sess", values="vals", icc_type='icc_1')
    assert np.allclose(round(icc, 2), -.05)

