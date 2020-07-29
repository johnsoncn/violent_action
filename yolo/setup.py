# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 15:50:25 2020

@author: nmc_costa
"""

import os
from setuptools import find_packages, setup

setup(
    name='yolo',
    version='0.1dev',
    long_description=open(os.path.join('yolov5','README.md')).read(),
    packages=find_packages()
)