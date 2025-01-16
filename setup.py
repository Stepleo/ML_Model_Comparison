from setuptools import setup, find_packages

setup(
    name="ClassComp", # for Classification comparison
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "matplotlib",
        "pandas",
    ],
)
