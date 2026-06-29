from setuptools import find_packages, setup

setup(
    name='active_rl_classification',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'gymnasium>=1.1.1',
        'scipy>=1.13.1',
        'stable-baselines3>=2.7.1',
        'torch>=2.8.0',
        'wandb>=0.25.1',
        'tensorboard',
        'matplotlib',
        'tqdm>=4.67.3',
    ],
)
