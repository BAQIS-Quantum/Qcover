import platform
from codecs import open
from os import path

#from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# def _process_requirements():
#     packages = open('requirements.txt', "r",  encoding="utf-16").read().strip().split('\n')
#     requires = []
#     for pkg in packages:
#         if pkg.startswith('git+ssh'):
#             return_code = os.system('pip install {}'.format(pkg))
#             assert return_code == 0, 'error, status_code is: {}, exit!'.format(return_code)
#         else:
#             requires.append("\"" + str(pkg) + "\"")
#     return requires

# This reads the __version__ variable from Qcover/version.py
__version__ = ""
exec(open('Qcover/version.py').read())

requirements = [
    "numpy>=1.19.3",
    "networkx>=2.5.1",
    "qiskit==0.31.0",
    "projectq==0.6.1.post0",
    "cirq==0.13.0",
    "quimb==1.3.0",
    "qulacs==0.3.0",
    "pyquafu>=0.2.4"
]

setup(
    name="Qcover",
    version=__version__,
    author="ntyz&finleyzhuang",
    author_email="puyn@baqis.ac.cn",
    description="Quantum computing solver",
    long_description=long_description,
    long_description_content_type='text/markdown',
    license="Apache-2.0 License",
    url="https://github.com/BAQIS-Quantum/Qcover",
    keywords="QAOA based combinational optimization solver QUBO quantum",

    packages=find_packages(),
    #ext_modules = cythonize("waveforms/math/prime.pyx"),
    include_package_data=True,
    #data_files=[('waveforms/Data', waveData)],
    install_requires=requirements,
    extras_require={
        'test': [
            'pytest>=4.4.0',
        ],
        'docs': [
            'Sphinx',
            'sphinxcontrib-napoleon',
            'sphinxcontrib-zopeext',
        ],
    },
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Topic :: Scientific/Engineering :: Interface Engine/Protocol Translator',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/BAQIS-Quantum/Qcover/issues',
        'Source': 'https://github.com/BAQIS-Quantum/Qcover',
    },
)