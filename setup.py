from setuptools import setup

setup(
    name='SafeRLBench',
    version='0.1',
    author='Nicolas Ochsner',
    author_email='ochsnern (at) student.ethz.ch',
    packages=['SafeRLBench'],
    description='Safe Reinforcement Learning Benchmark',
    install_requires=[
        'numpy >= 1.7',
        'six >= 1.10'
    ]
)
