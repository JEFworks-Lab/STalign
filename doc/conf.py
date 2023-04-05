# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'STalign'
copyright = '2023, JEFworks Lab'
author = 'Manjari Anant, Jean Fan'

release = '0.1'
version = '0.1.0'

# -- General configuration

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['myst_parser', 'sphinx.ext.autodoc', 'autodocsumm', 'nbsphinx',
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
