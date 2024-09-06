from setuptools import setup, find_packages

setup(
    name="repana",
    version="0.1.2",
    packages=find_packages(),
    install_requires=[
        "transformers", "torch", "numpy", "tqdm", "scikit-learn"
    ]
)