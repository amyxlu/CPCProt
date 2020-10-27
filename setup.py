import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CPCProt",
    version="0.0.1",
    author="Amy Lu",
    author_email="amyxlu@cs.toronto.edu",
    description="Parameter-efficient pretrained model for obtaining protein embeddings.",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.19,<2.0",
        "torch>=1.0,<2.0",
        "lmdb>=1.0,<2.0",
        # "tape-proteins>=0.4,<=1.0",
        "tape-proteins @ git+https://github.com/konstin/tape@patch-1",
        "scipy>=1.5,<2.0",
    ],
)
