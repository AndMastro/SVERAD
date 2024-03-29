from setuptools import setup
from setuptools import find_packages

setup(
    name='SVERAD',
    version='1.0.1',
    author='Andrea Mastropietro',
    license="MIT",
    #packages = find_packages('src'),
    packages=['sverad'],
    #package_dir={'sverad': 'src'},
    author_email='mastropietro@diag.uniroma1.it',
    url='https://github.com/AndMastro/SVERAD',
    description='Method to perform exact Shapley value computation for SVM models with the RBF kernel using binary fingerprints.',
    install_requires=['numpy', 'matplotlib' , 'scikit-learn', 'scipy', 'rdkit', 'bidict'] 
)