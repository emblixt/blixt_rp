from setuptools import setup
import re

verstr = 'unknown'
VERSIONFILE = "blixt_rp/_version.py"
with open(VERSIONFILE, "r") as f:
    verstrline = f.read().strip()
    pattern = re.compile(r"__version__ = ['\"](.*)['\"]")
    mo = pattern.search(verstrline)
if mo:
    verstr = mo.group(1)
    print("Version "+verstr)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setup(
    name='blixt_rp',
    version=verstr,
    packages=[
        'blixt_rp',
        'blixt_rp.core',
        'blixt_rp.plotting',
        'blixt_rp.rp',
        'blixt_rp.rp_utils',
        'blixt_rp.unit_tests'
    ],
    url='https://github.com/emblixt/blixt_rp',
    license='GNU 3.0',
    author='Erik Mårten Blixt',
    author_email='marten.blixt@gmail.com',
    description='Some scripts handy for analysing wells, doing basic rock physics and AVO modelling',
    long_description='',
    zip_safe=False,
    platforms='any',
    install_requires=[
        'numpy>=1.16.0',
        'matplotlib>=3.0.2',
        'scipy>=1.4.1',
        're>=2.2.1',
        'json>=2.0.9',
        'logging>=0.5.1.2',
        'pandas>=1.1.0',
        'pywt>=1.1.1',
        'setuptools>=47.1.0',
        'blixt_utils>=0.1.0'

    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
    ],
)
