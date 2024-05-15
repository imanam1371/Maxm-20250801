from distutils.core import setup
from glob import glob
module = 'MAXM'
py_modules = glob(module)
py_modules = [x.replace('.py', '') for x in py_modules if "__" not in x]
setup(
    name=f'{module}_lib',
    version='1.0',
    description=f'Python utilities for the {module}',
    packages=[module],
    py_modules=py_modules,
)
