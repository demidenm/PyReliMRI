import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from imgreliability.calc_image_similarity import (
    image_similarity,
    sumsq_total,
    sumsq_within,
    sumsq_btwn,
    calculate_icc
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

# test ICC calc is close for specified r
def create_corrdat(r):
    subj = np.arange(1, 1000 + 1, 1)
    cov = np.array([[1., r],
                    [r, 1.]])

    np.random.seed(111)
    repeated = multivar_norm.rvs(mean=[0, 0],
                                 cov=cov, size=1000)

    df = pd.DataFrame(data=np.column_stack((subj, repeated)),
                      columns=["subj", "A1", "A2"])

    return df

@pytest.mark.parametrize("corr", [.50, .60,.45])
def test_calculate_icc(corr):
    df = create_corrdat(corr)
    icc = calculate_icc(df_wide=df, sub_var="subj",
                        sess_vars=["A1", "A2"],
                        icc_type='icc_3')

    assert np.allclose(round(icc, 2), corr)

# test degenerate case where ssqt,within,between is zero
def test_sumsq_total():
    data = np.ones(100)
    ssqt = sumsq_total(data)
    assert np.allclose(ssqt, 0)

def test_sumsq_total():
    n = 100
    subj = np.tile(np.arange(1, n + 1, 1), 2)
    measures = np.tile(np.ones(n), 2)
    df_lg = pd.DataFrame(data=np.column_stack((subj, measures)),
                         columns=["subj", "values"])
    ssqt = sumsq_total(df_long=df_lg, values="values")
    assert np.allclose(ssqt, 0)


def test_sumsq_within():
    n_subj = 100
    n_sess = 2
    subj = np.tile(np.arange(1, n_subj + 1, 1), 2)
    measures = np.tile(np.ones(n_subj), 2)
    sessions = np.hstack((np.ones(n_subj), np.zeros(n_subj)))
    df_lg = pd.DataFrame(data=np.column_stack((subj, sessions, measures)),
                         columns=["subj", "sess", "values"])
    ssq_wthn = sumsq_within(df_long=df_lg, intra_var="sess",
                            values="values", n_subjects=n_subj)

    assert np.allclose(ssq_wthn, 0)


def test_sumsq_btwn():
    n_subj = 100
    n_sess = 2
    subj = np.tile(np.arange(1, n_subj + 1, 1), 2)
    measures = np.tile(np.ones(n_subj), 2)
    sessions = np.hstack((np.ones(n_subj), np.zeros(n_subj)))
    df_lg = pd.DataFrame(data=np.column_stack((subj, sessions, measures)),
                         columns=["subj", "sess", "values"])
    sssq_btwn = sumsq_btwn(df_long=df_lg, inter_var="subj",
                           values="values", n_sessions=n_sess)

    assert np.allclose(sssq_btwn, 0)


