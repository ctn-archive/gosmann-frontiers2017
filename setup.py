from setuptools import find_packages, setup

print(find_packages() + [
        'gosmann_frontiers2016.' + p for p in find_packages('spaun2.0')])

setup(
    name="gosmann_frontiers2016",
    version="1.0",
    author="Jan Gosmann",
    author_email="jgosmann@uwaterloo.ca",
    packages=find_packages() + [
        'gosmann_frontiers2016.' + p for p in find_packages('spaun2.0')],
    package_dir={'gosmann_frontiers2016._spaun': 'spaun2.0/_spaun'},
    package_data={
        'gosmann_frontiers2016._spaun.modules.vision': ['*.npz', '*.gz', '*.pkl'],
        'gosmann_frontiers2016._spaun.modules.motor': ['*.npz', '*.gz', '*.pkl']},
)
