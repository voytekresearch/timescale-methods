# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# For a full list of documentation options, see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# ----------------------------------------------------------------------------

import os
from os.path import dirname as up

from datetime import date

import sphinx_book_theme
from timescales import __version__

# -- Project information -----------------------------------------------------

# Set project information
project = 'timescales'
copyright = '2021-{}, VoytekLab'.format(date.today().year)
author = 'Ryan Hammonds'

# Get and set the current version number

version = __version__
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.githubpages',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx_gallery.gen_gallery',
    'sphinx_copybutton',
    'sphinx_design',
    'sphinx_remove_toctrees',
    'numpydoc',
    'nbsphinx'
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

exclude_patterns = ['_build']

# numpydoc interacts with autosummary, that creates excessive warnings
# This line is a 'hack' for that interaction that stops the warnings
numpydoc_show_class_members = False

# generate autosummary even if no references
autosummary_generate = True

# The master toctree document.
master_doc = 'index'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# Settings for sphinx_copybutton
copybutton_prompt_text = "$ "

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = 'sphinx_book_theme'

# Theme options to customize the look and feel, which are theme-specific.
html_theme_options = {
    'repository_url': "https://github.com/voytekresearch/timescale-methods",
    'use_repository_button': True,
    "show_navbar_depth": 0,
    "show_nav_level": 0,
    "use_repository_button": True,
    "show_toc_level": 1
}

remove_from_toctrees = ["./generated/*", './figures/*']

# Settings for whether to copy over and show link rst source pages
html_copy_source = False
html_show_sourcelink = False

# -- Extension configuration -------------------------------------------------

# Configurations for sphinx gallery
sphinx_gallery_conf = {
    'examples_dirs': ['figures', 'tutorials'],
    'gallery_dirs': ['auto_figures', 'auto_tutorials'],
    'backreferences_dir': 'generated',
    'doc_module': ('timescales',),
    'reference_url': {'timescales': None},
    'remove_config_comments': True,
}

nb_execution_mode = 'off'
