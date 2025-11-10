from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'waypoint_pub'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')), #.pak all the file in launch here
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yuzhou',
    maintainer_email='root@todo.todo',
    description='Waypoint publisher for PX4',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'waypoint_fly = waypoint_pub.waypoint_fly:main',
        ],
    },
)
