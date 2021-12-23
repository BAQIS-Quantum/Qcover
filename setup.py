from distutils.core import setup
import setuptools
import os

with open("README.md", "r") as fh:
  long_description = fh.read()

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

setuptools.setup(
  name="Qcover",
  version="1.0.0",
  author="ntyz&finleyzhuang",
  author_email="puyn@baqis.ac.cn",
  description="Quantum computing solver",
  long_description=long_description,
  long_description_content_type="text/markdown",
  license="Apache-2.0 License",
  url="https://github.com/BAQIS-Quantum/Qcover",
  packages=setuptools.find_packages(),
  # packages=setuptools.find_packages('src'),
  # package_dir={'':'src'},
  # include_package_data=True,

  # install_requires=_process_requirements(),
  install_requires=[
  ],
  classifiers=[
  "Programming Language :: Python :: 3.8",
  "Operating System :: OS Independent",
  ],
)

# "networkx=2.5.1=pypi_0",
# "qiskit=0.31.0=pypi_0",
# 'qiskit-aer=0.9.1=pypi_0',
# 'qiskit-aqua=0.9.5=pypi_0',
# 'qiskit-ibmq-provider=0.17.0=pypi_0',
# 'qiskit-ignis=0.6.0=pypi_0',
# 'qiskit-terra=0.18.3=pypi_0',
# "projectq=0.6.1.post0=pypi_0",
# "cirq=0.13.0=pypi_0",
# "cirq-aqt=0.13.0=pypi_0",
# "cirq-core=0.13.0=pypi_0",
# "cirq-google=0.13.0=pypi_0",
# "cirq-ionq=0.13.0=pypi_0",
# "cirq-pasqal=0.13.0=pypi_0",
# "cirq-rigetti=0.13.0=pypi_0",
# "cirq-web=0.13.0=pypi_0",
# "quimb=1.3.0+345.gf06427e=pypi_0",
# "qulacs=0.3.0=pypi_0",