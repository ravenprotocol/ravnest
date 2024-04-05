import sys
import os

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Ravnest'
copyright = '2024, Raven Protocol'
author = 'Raven Protocol'

version = 'latest'
github_url = 'https://github.com/ravenprotocol/ravnest'
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 
              'myst_parser', 
              'sphinx_copybutton',
              'sphinx_rtd_theme']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_mock_imports = ['torch',
                        'grpcio',
                        'grpc',
                        'grpcio-tools',
                        'numpy',
                        'protobuf',
                        'scikit_learn',
                        'torchvision',
                        'torchpippy',
                        'pippy',
                        'packaging',
                        'psutil',
                        'ravnest.protos.server_pb2_grpc', 
                        'ravnest.protos.tensor_pb2',
                        'ravnest.protos.server_pb2']

# pygments_style = 'monokai'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'display_version': True,
    'collapse_navigation': False
}
html_favicon = "_static/raven-white.png"
html_static_path = ['_static']
