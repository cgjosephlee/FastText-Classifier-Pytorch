from setuptools import setup, find_packages

setup(
    name="fasttext-classifier",
    version="0.0.2",
    url="https://github.com/cgjosephlee/FastText-Classifier-Pytorch",
    author="Joseph Lee",
    author_email="cgjosephlee@gmail.com",
    description="FastText classifier in Pytorch",
    packages=find_packages(),
    install_requires=[
        # "torch",
        "pytorch-lightning",
        "gensim"
        ],
)
