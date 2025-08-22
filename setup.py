import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyReliMRI",
    version="2.2.0",
    description="A package for computing reliability of MRI/fMRI images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Michael Demidenko",
    author_email="demidenko.michael@gmail.com",
    url="https://github.com/demidenm/PyReliMRI",
    project_urls={
        "Bug Tracker": "https://github.com/demidenm/PyReliMRI/issues",
    },
    license="MIT",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy", 
        "pandas",
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
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    include_package_data=True,
)