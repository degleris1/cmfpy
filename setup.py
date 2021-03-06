try:
    from setuptools import setup, find_packages
except ImportError:
    from disutils.core import setup, find_packages

import cmfpy

config = {
    'name': 'cmfpy',
    'packages': find_packages(exclude=['doc']),
    'description': 'Tools for Convolutive Matrix Factorization',
    'author': 'Anthony Degleris, Alex Williams, Ben Antin',
    'author_email': 'degleris@stanford.edu',
    'url': 'https://github.com/degleris1/cmfpy',
}

setup(**config)
