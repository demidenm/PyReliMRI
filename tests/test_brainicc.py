import pytest
import os
import numpy as np
import nibabel as nib
from pathlib import Path
from pyrelimri.similarity import image_similarity
from pyrelimri.brain_icc import (voxelwise_icc, setup_atlas, roi_icc)
from nilearn.datasets import fetch_neurovault_ids
from collections import namedtuple
from nilearn.masking import compute_multi_brain_mask


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
            maskdims = [round(i / 2) for i in imgdims]
        mask = np.zeros(imgdims).astype('int')
        mask[:maskdims[0], :maskdims[1], :maskdims[2]] = 1
    else:
        mask = np.ones(imgdims)

    maskvox = mask.reshape(nvox)

    empirical_r = 10
    while (np.abs(empirical_r - r) > tol):
        data = rng.multivariate_normal(mean=[0, 0],
                                       cov=[[1, r], [r, 1]], size=nvox)
        empirical_r = np.corrcoef(data[maskvox == 1, :].T)[0, 1]

    for i in range(2):
        imgpair.images.append(dir / f'testimg_{i}.nii.gz')
        tmpimg = nib.Nifti1Image((maskvox * data[:, i]).reshape(imgdims),
                                 affine=np.eye(4))
        tmpimg.to_filename(imgpair.images[-1])
    imgpair.mask = dir / 'mask.nii.gz'
    maskimg = nib.Nifti1Image(mask, affine=np.eye(4))
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


def create_dummy_nifti(shape, affine, filepath):
    data = np.random.rand(*shape)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(filepath))


def test_session_lengths_mismatch(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("data")

    # Test case with different session lengths
    multisession_list = [
        [tmpdir / "sub-00_ses1_Contrast-A_bold.nii.gz",
         tmpdir / "sub-01_ses1_Contrast-A_bold.nii.gz"],
        [tmpdir / "sub-00_ses2_Contrast-A_bold.nii.gz",
         tmpdir / "sub-01_ses2_Contrast-A_bold.nii.gz",
         tmpdir / "sub-03_ses2_Contrast-A_bold.nii.gz"]
    ]

    icc_type = "icc_3"

    # Create dummy NIfTI files
    shape = (97, 115, 97)
    affine = np.eye(4)
    for session in multisession_list:
        for filepath in session:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            create_dummy_nifti(shape, affine, filepath)

    mask = compute_multi_brain_mask(target_imgs=[
        tmpdir / "sub-00_ses2_Contrast-A_bold.nii.gz",
        tmpdir / "sub-01_ses2_Contrast-A_bold.nii.gz",
        tmpdir / "sub-03_ses2_Contrast-A_bold.nii.gz"
    ])

    mask_path = tmpdir / 'test_mask.nii.gz'
    nib.save(mask, mask_path)

    # The assertion should raise an exception
    with pytest.raises(AssertionError):
        voxelwise_icc(multisession_list, mask, icc_type)


@pytest.mark.parametrize("measure", ['Dice', 'Jaccard'])
def test_image_similarity(image_pair, measure):
    imgsim = image_similarity(
        image_pair.images[0], image_pair.images[1], image_pair.mask,
        thresh=1, similarity_type=measure
    )
    assert imgsim is not None


def test_spearman_similarity(image_pair):
    spearman_sim = image_similarity(
        image_pair.images[0], image_pair.images[1], similarity_type='spearman')
    assert abs(spearman_sim - 0.5) < 0.1, "The similarity is not close to 0.5."


def test_image_similarity_spearman_value_error(image_pair):
    imgfile1 = image_pair.images[0]
    imgfile2 = image_pair.images[1]
    mask = None
    thresh = 0.5
    similarity_type = "spearman"
    with pytest.raises(ValueError):
        image_similarity(imgfile1, imgfile2, mask=mask, thresh=thresh, similarity_type=similarity_type)


# test roi based ICC
def setup_atlas_valuerrror():
    with pytest.raises(ValueError):
        setup_atlas(name_atlas='fake_atlas')

@pytest.mark.parametrize("atlases", ['aal', 'difumo'])
def setup_atlas_noerror(atlases):
    setup_atlas(name_atlas=atlases)

def test_roiicc_msc(tmp_path_factory):

    # create temp dir
    tmpdir = tmp_path_factory.mktemp("data")

    # Test case w/ neurovault data
    MSC01_1 = fetch_neurovault_ids(image_ids=[48068], data_dir=tmpdir)  # MSC01 motor session1 1 L Hand beta
    MSC01_2 = fetch_neurovault_ids(image_ids=[48073], data_dir=tmpdir)  # MSC01 motor session2 1 L Hand beta
    MSC02_1 = fetch_neurovault_ids(image_ids=[48118], data_dir=tmpdir)
    MSC02_2 = fetch_neurovault_ids(image_ids=[48123], data_dir=tmpdir)
    MSC03_1 = fetch_neurovault_ids(image_ids=[48168], data_dir=tmpdir)
    MSC03_2 = fetch_neurovault_ids(image_ids=[48173], data_dir=tmpdir)

    ses1 = [MSC01_1['images'], MSC02_1['images'], MSC03_1['images']]
    ses2 = [MSC01_2['images'], MSC02_2['images'], MSC03_2['images']]

    # estimate ICC for roi = 200 in shaefer
    result = roi_icc(multisession_list=[ses1, ses2], type_atlas='shaefer_2018',
                     atlas_dir=tmpdir, icc_type='icc_3')

    assert np.allclose(result['est'][200], .70, atol=.01)