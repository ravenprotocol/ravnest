import subprocess
import sys
from pathlib import Path
import glob
from pkg_resources import parse_requirements
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def proto_compile(output_path=this_directory):
    install("grpcio-tools==1.51.3")
    import grpc_tools.protoc

    cli_args = [
        "grpc_tools.protoc",
        "--python_out={}".format(output_path),
        "--pyi_out={}".format(output_path),
        "--grpc_python_out={}".format(output_path),
    ] + glob.glob("protos/*.proto")

    code = grpc_tools.protoc.main(cli_args)
    

class BuildPy(build_py):
    def run(self):
        super().run()
        proto_compile(this_directory)


with open("requirements.txt") as requirements_file:
    install_requires = list(map(str, parse_requirements(requirements_file)))

setup(
    name="ravnest",
    version="0.1.0",
    cmdclass={"build_py":BuildPy},
    license='MIT',
    author="Raven Protocol",
    author_email='kailash@ravenprotocol.com',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ravenprotocol/ravnest',
    keywords='deep learning, distributed computing, decentralized training',
    install_requires=install_requires
)