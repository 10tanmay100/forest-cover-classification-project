from gettext import install
from setuptools import setup
from typing import List


def get_requirements_list()->List[str]:
    '''
    This function will going to return a list of strings which will contain name for all the libraries specified with requirements.txt
    '''
    with open("requirements.txt","r") as f:
        return f.readlines().remove("-e .")

setup(
name="forest-cover-predictor",
version="0.0.2",
author="Tanmay",
description="This is cover type predictor setup file",
packages=["forest_cover"],
install_requires=get_requirements_list()
)



