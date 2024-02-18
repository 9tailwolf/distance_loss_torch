from setuptools import setup, find_packages

setup(
    name='distance_loss_torch',
    version='0.0.6',
    description='Distance Loss function package for pytorch',
    author='9tailwolf',
    author_email='doryeon514@gmail.com',
    url='https://github.com/9tailwolf/distance_loss_torch',
    install_requires=['torch',],
    packages=find_packages(exclude=[]),
    keywords=['loss funtion','deep learning','pytorch'],
    python_requires='>=3.6',
    package_data={},
    zip_safe=False,
)