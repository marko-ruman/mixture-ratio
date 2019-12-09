from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='MixtureRatio',
      version='0.0.00',
      url='https://github.com/marko-ruman/mixture-ratio',
      license='MIT',
      author='Marko Ruman',
      author_email='marko.ruman@gmail.com',
      description='Implementation of Mixture Ratio probabilistic model',
      packages=find_packages(),
      # long_description=long_description,
      zip_safe=False)