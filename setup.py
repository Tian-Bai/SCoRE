from setuptools import setup, find_packages

setup(
    name='SCoRE_pkg',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn',
        'tqdm'
    ],
)
