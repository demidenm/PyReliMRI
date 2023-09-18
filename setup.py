import setuptools

setuptools.setup(
    name="PyReliMRI",
    version="1.1.0",
    description="A package for computing reliability of fMRI images",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "pytest",
        "nilearn",
        "nibabel",
        "scipy",
        "seaborn",
        "scikit-learn",
        "hypothesis"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)