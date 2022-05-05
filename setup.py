from setuptools import setup, find_packages

setup(
    name='graphai',
    version='0.1.0',
    description='CEDE EPFL Graph AI',
    author='Aitor Pérez',
    author_email='aitor.perez@epfl.ch',
    url='',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'uvicorn',
        'fastapi',
        'elasticsearch',
        'numpy',
        'pandas',
        'matplotlib',
        'python-rake',
        'requests-futures',
        'nltk',
        'rake-nltk',
        'wikipedia',
        'ray',
        'python-Levenshtein',
        'mysql-connector',
        'mwparserfromhell'
    ],
    extras_require={
        'docs': ['sphinx']
    },
    license=''
)