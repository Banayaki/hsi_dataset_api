from setuptools import setup
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='HSI_Dataset_API',
    packages=setuptools.find_packages(),
    version='1.4.6',
    description='API for accessing HSI datasets',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Mukhin Artem',
    author_email='artemmukhinssau@gmail.com',
    url='https://github.com/Banayaki/hsi_dataset_api',
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ], install_requires=[]
)
