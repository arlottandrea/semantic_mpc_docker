## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=[
        'semantic_mpc_package',
        'semantic_mpc_package.baselines',
        'semantic_mpc_package.nodes',
        'semantic_mpc_package.ros_com_lib',
    ],
    package_dir={'': 'src'},
    package_data={'semantic_mpc_package': ['models/*.pth', 'models/*/*.pth']},
)

setup(**setup_args)
