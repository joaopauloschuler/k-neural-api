from setuptools import setup

long_description = '''
K-CAI NEURAL API is a Keras based neural network API that will allow you to prototype faster!
This project is a subproject from a bigger and older project called CAI and is sister to the pascal based CAI NEURAL API.
'''

setup(name='cai',
      version='0.1.7',
      description='K-CAI NEURAL API',
      long_description=long_description,
      author='Joao Paulo Schwarz Schuler',
      url='https://github.com/joaopauloschuler/k-neural-api',
      install_requires=[ # 'tensorflow>=2.2.0', # leaving this as a require sometimes replaces tensorflow
                        'pandas>=0.22.0',
                        'scikit-image>=0.15.0',
                        'opencv-python>=4.1.2.30', 
                        'scikit-learn>=0.21.0',
                        'numpy'],
      classifiers=[
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=['cai'])
