from setuptools import find_packages, setup

setup(
    name="gosmann_frontiers2016",
    version="1.0",
    author="Jan Gosmann",
    author_email="jgosmann@uwaterloo.ca",
    packages=find_packages() + ['gosmann_frontiers2016._spaun'],
    package_dir={'gosmann_frontiers2016._spaun': 'spaun2.0/_spaun'},
)
