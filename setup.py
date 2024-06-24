import setuptools

setuptools.setup(
    name="PyReliMRI",
    version="2.1.0",
    description="A package for computing reliability of MRI/fMRI images",
    author="Michael Demidenko",
    author_email="demidenko.michael@gmail.com",
    url="https://github.com/demidenm/PyReliMRI",
    license="MIT",
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
        "hypothesis",
        "matplotlib",
        "joblib",
        "statsmodels",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
