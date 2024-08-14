from setuptools import setup

with open('requirements.txt') as f:
    install_reqs = f.read().splitlines()

setup(
      name = 'QMy',
      version = '0.1',
      description = 'Scripts to calculate the Q and M variability metrics according to Robinson et al. (2021).',
      author = 'John Carlos Wendeborn',
      author_email = 'jwendeborn@gmail.com',
      install_requires = install_reqs,
      license = 'MIT',
      url = 'https://github.com/wendeborn8/QMy',
      
      )