from distutils.core import setup
setup(
  name = 'hierreg',
  packages = ['hierreg'],
  install_requires=[
   'pandas',
   'numpy',
   'scipy',
   'casadi',
   'cvxpy'
],
  version = '0.1',
  description = 'Hierarchical or multilevel linear models through l2-regularization',
  author = 'David Cortes',
  author_email = 'david.cortes.rivera@gmail.com',
  url = 'https://github.com/david-cortes/hierreg',
  download_url = 'https://github.com/david-cortes/hierreg/archive/0.1.tar.gz',
  keywords = ['random effects', 'mixed effects', 'herarchical model', 'multilevel modeling'],
  classifiers = [],
)