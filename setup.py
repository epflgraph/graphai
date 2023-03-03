from setuptools import setup, find_packages

setup(
    name='graphai',
    version='0.2.0',
    description='EPFL Graph AI',
    author='Aitor Pérez',
    author_email='aitor.perez@epfl.ch',
    url='graphai.epfl.ch',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'fastapi',
        'uvicorn',
        'pydantic',
        'requests',
        'ray',
        'mysql-connector-python',
        'elasticsearch',
        'unidecode',
        'clean-text',
        'python-rake',
        'nltk',
        'rake-nltk',
        'levenshtein',
        'mwparserfromhell',
        'sphinx',
        'sphinx-material'
    ]
)

