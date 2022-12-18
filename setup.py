from setuptools import setup, find_packages

setup(
    name="fasttext-classifier",
    version="0.0.1",
    url="https://github.com/cgjosephlee/FastText-Classifier-Pytorch",
    author="Joseph Lee",
    author_email="cgjosephlee@gmail.com",
    description="FastText classifier in Pytorch",
    packages=find_packages(),
    # package_data={"mpl_no_tofu": ["config.json", "assets/*/METADATA.pb"]},
    install_requires=[
        "pytorch",
        "pytorch-lightning",
        "gensim"
        ],
)
