# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath("../../keras_mml"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Keras-MatMulLess"
copyright = "2024, PhotonicGluon"
author = "PhotonicGluon"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # Sphinx's own extensions
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.viewcode",
    # External extensions
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_tippy",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".md": "myst-nb",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_js_files = ["js/version-alert.js"]

html_theme_options = {
    "source_repository": "https://github.com/PhotonicGluon/Keras-MatMulLess",
    "source_branch": "main",
    "source_directory": "docs/source",
}

# -- Options for extensions --------------------------------------------------
doctest_global_setup = """
import numpy as np
"""

autosectionlabel_prefix_document = True

napoleon_include_init_with_doc = True
napoleon_numpy_docstring = False
napoleon_use_rtype = False

autodoc_class_signature = "separated"

typehints_document_rtype = True
typehints_use_rtype = False
typehints_defaults = "comma"

intersphinx_disabled_domains = ["std"]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "keras": ("https://keras.io/api/", "objects_keras.inv"),
}

autosummary_ignore_module_all = False

myst_enable_extensions = ["colon_fence", "fieldlist", "dollarmath", "attrs_block"]
myst_dmath_double_inline = True
myst_words_per_minute = 100

nb_execution_mode = "off"
