from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("tree", ["tree.pyx"])]
    #ext_modules = [Extension("brute_force_3d", ["brute_force_3d.pyx"])]
)
