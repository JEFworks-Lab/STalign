# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os
sys.path.insert(0,os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'STalign'
copyright = '2023, JEFworks Lab'
author = 'Manjari Anant, Kalen Clifton, Daniel Tward, Jean Fan, et al.'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['myst_parser', 'autodocsumm', 'nbsphinx', 
]
source_suffix = ['.rst', '.md'
]

autodoc_default_options = {
	'autosummary': True
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = 'https://github.com/JEFworks-Lab/STalign/blob/ede36ce5ce3f7151251c271a56811b31c7bde908/STalign_logos_fin.png'

