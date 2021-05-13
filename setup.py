import os
from setuptools import setup

if os.environ.get("READTHEDOCS", False) == "True":
    INSTALL_REQUIRES = []
else:
    INSTALL_REQUIRES = ['matplotlib']
    

setup(name='plots',
    version='0.1',
    description='Matplotlib abstraction to make plotting easier',
    url='https://github.com/NiklasPhabian/plots',
    author='Niklas Griessbaum',      
    license='MIT', 
    py_modules=['plots'],
    zip_safe=False,
    install_requires=INSTALL_REQUIRES
)
