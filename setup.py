from setuptools import setup, find_packages

setup(name='mixture-ratio',
      version='0.0.00',
      url='https://github.com/marko-ruman/mixture-ratio',
      license='MIT',
      author='Marko Ruman',
      author_email='marko.ruman@gmail.com',
      description='Implementation of novel Mixture Ratio probabilistic model',
      packages=find_packages(),
      long_description=open('README.md').read(),
      zip_safe=False)