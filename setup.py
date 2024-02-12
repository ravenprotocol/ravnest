import subprocess
import sys
from pathlib import Path
import glob
from pkg_resources import parse_requirements
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from setuptools.command.build_py import build_py

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def proto_compile(output_path=this_directory):
    install_package("grpcio-tools==1.51.3")
    import grpc_tools.protoc
    print('Output Path: ', output_path)
    cli_args = [
        "grpc_tools.protoc",
        "--proto_path=ravnest/protos",
        "--python_out={}/ravnest/protos".format(output_path),
        "--pyi_out={}/ravnest/protos".format(output_path),
        "--grpc_python_out={}/ravnest/protos".format(output_path),
    ] + glob.glob("ravnest/protos/*.proto")
    print('GLOB: ', glob.glob("ravnest/protos/*.proto"))
    code = grpc_tools.protoc.main(cli_args)
    

class CustomInstallCommand(install):
    def run(self):
        super().run()
        proto_compile(this_directory)

class CustomDevelopCommand(develop):
    def run(self):
        super().run()
        proto_compile(this_directory) 

class CustomBuildpyCommand(build_py):
    def run(self):
        super().run()
        proto_compile(this_directory)       

class CustomEggInfoCommand(egg_info):
    def run(self):
        super().run()
        proto_compile(this_directory)

with open("requirements.txt") as requirements_file:
    install_requires = list(map(str, parse_requirements(requirements_file)))

setup(
    name="ravnest",
    version="0.1.0",
    cmdclass={"install":CustomInstallCommand, 
              "build_py":CustomBuildpyCommand,
              "develop":CustomDevelopCommand,
              "egg_info":CustomEggInfoCommand},
    license='MIT',
    author="Raven Protocol",
    author_email='kailash@ravenprotocol.com',
    packages=find_packages(),
    package_data={"ravnest":["protos/*"]},
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ravenprotocol/ravnest',
    keywords='deep learning, distributed computing, decentralized training',
    install_requires=install_requires
)