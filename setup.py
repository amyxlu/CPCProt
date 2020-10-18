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
    python_requires='>=3.6',
)
