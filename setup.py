from setuptools import setup
import os
from glob import glob

package_name = 'ft_semantic_segmentation'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, 'ft_semantic_segmentation/models', 'ft_semantic_segmentation/models/base_models', 'ft_semantic_segmentation/utils', 'ft_semantic_segmentation/nn', 'ft_semantic_segmentation/data', 'ft_semantic_segmentation/data/dataloader'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='naza',
    maintainer_email='nozolo90@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
