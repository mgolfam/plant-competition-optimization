from setuptools import setup, find_packages

setup(
    name='pco-algorithm',
    version='1.0.0',
    author='Milad Golfam',
    description='Implementation of Plant Competition Optimization, PSO, and ACO algorithms.',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'numpy',
        'matplotlib'
    ],
)
