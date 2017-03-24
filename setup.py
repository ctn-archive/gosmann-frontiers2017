from setuptools import find_packages, setup


setup(
    name="gosmann_frontiers2017",
    version="1.1",
    author="Jan Gosmann",
    author_email="jgosmann@uwaterloo.ca",
    packages=find_packages() + [
        'gosmann_frontiers2017.' + p for p in find_packages('spaun2.0')],
    package_dir={'gosmann_frontiers2017._spaun': 'spaun2.0/_spaun'},
    package_data={
        'gosmann_frontiers2017._spaun.modules.vision': [
            '*.npz', '*.gz', '*.pkl'],
        'gosmann_frontiers2017._spaun.modules.motor': [
            '*.npz', '*.gz', '*.pkl']},
    install_requires=[
        'matplotlib',
        'nengo >= 2.3.0',
        'nengo_ocl >= 1.0.0',
        'numpy>=1.10',
        'psutil',
        'psyrun>=0.5.3',
        'scipy',
    ],
)
