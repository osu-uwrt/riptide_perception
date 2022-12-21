from setuptools import setup, find_packages

import os
from glob import glob
from urllib.request import urlretrieve

package_name = 'riptide_yolo'

packages = list(find_packages())
packages.append(package_name)
modules = []
for pack in packages:
    dirName = str(pack).replace('.', '/')
    modules.extend(glob(dirName + '/[!_]*.py'))

cleanModules = []
for module in modules:
    clean = str(module).replace('/', '.')[:-3]
    cleanModules.append(clean)

setup(
    name=package_name,
    version='0.2.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'data'), glob('data/*.yaml')),
        (os.path.join('share', package_name, 'weights'), glob('weights/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Ar-Ray-code',
    author_email="ray255ar@gmail.com",
    maintainer='Ar-Ray-code',
    maintainer_email="ray255ar@gmail.com",
    description='YOLOv5 + ROS2',
    license='GNU GENERAL PUBLIC LICENSE Version 3',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vision = riptide_yolo.vision:ros_main'
        ],
    },
    py_modules=cleanModules
)