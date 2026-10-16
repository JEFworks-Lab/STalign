# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os
sys.path.insert(0,os.path.abspath('..'))
import STalign  # noqa: E402  (this checkout's package, for its version)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'STalign'
copyright = '2026, JEFworks Lab'
author = 'Manjari Anant, Kalen Clifton, Kamil Slowikowski, Daniel Tward, Jean Fan, et al.'
release = STalign.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['myst_parser', 'autodocsumm', 'nbsphinx',
              'sphinx.ext.napoleon',  # the docstrings are in numpy style
]
# anchors for the Markdown headings, so that links like README's #performance resolve
myst_heading_anchors = 3
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
html_logo = 'https://raw.githubusercontent.com/JEFworks-Lab/STalign/main/STalign_logos_fin.png'

