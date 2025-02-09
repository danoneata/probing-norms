from setuptools import setup, find_packages

setup(
    name="probing_norms",
    version="0.0.1",
    url="https://github.com/danoneata/probing-norms",
    author="Dan Oneață",
    author_email="dan.oneata@gmail.com",
    description="Understanding deep learning models by probing feature norms",
    packages=find_packages(),
    install_requires=["black", "click", "streamlit", "ruff", "tqdm", "transformers", "timm", "scikit-learn", "toolz", "h5py", "fasttext"],
)
