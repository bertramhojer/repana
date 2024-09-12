from setuptools import setup, find_packages

# Read requirements from file
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="repana",
    version="0.1.4",
    packages=find_packages(),
    install_requires=required
)