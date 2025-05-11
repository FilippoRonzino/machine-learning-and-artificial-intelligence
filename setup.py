from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='ai_project',
    version='0.1',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
)