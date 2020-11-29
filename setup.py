#!/usr/bin/env python

from setuptools import setup

setup(name='Erlich',
      version='0.1',
      description='TODO',
      author='Matteo Ronchetti',
      author_email='mttronchetti@gmail.com',
      url='https://github.com/matteo-ronchetti/erlich',
      packages=['erlich', "erlich.components"],
      install_requires=[
          'omegaconf',
      ],
)