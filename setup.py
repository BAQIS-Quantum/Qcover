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

# This reads the __version__ variable from waveforms/version.py
__version__ = ""
exec(open('qcover/version.py').read())

requirements = [
    "networkx==2.5.1=pypi_0",
    "qiskit==0.31.0=pypi_0",
    'qiskit-aer==0.9.1=pypi_0',
    'qiskit-aqua==0.9.5=pypi_0',
    'qiskit-ibmq-provider==0.17.0=pypi_0',
    'qiskit-ignis==0.6.0=pypi_0',
    'qiskit-terra==0.18.3=pypi_0',
    "projectq==0.6.1.post0=pypi_0",
    "cirq==0.13.0=pypi_0",
    "cirq-aqt==0.13.0=pypi_0",
    "cirq-core==0.13.0=pypi_0",
    "cirq-google==0.13.0=pypi_0",
    "cirq-ionq==0.13.0=pypi_0",
    "cirq-pasqal==0.13.0=pypi_0",
    "cirq-rigetti==0.13.0=pypi_0",
    "cirq-web==0.13.0=pypi_0",
    "quimb==1.3.0+345.gf06427e=pypi_0",
    "qulacs==0.3.0=pypi_0",
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
    keywords="QAOA based combinational optimization solver",

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
        'License :: OSI Approved :: Apache-2.0 License',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Topic :: Scientific/Engineering :: Interface Engine/Protocol Translator',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/BAQIS-Quantum/Qcover/issues',
        'Source': 'https://github.com/BAQIS-Quantum/Qcover',
    },
)