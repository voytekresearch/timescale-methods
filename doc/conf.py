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

import sphinx_gallery
import sphinx_book_theme
from sphinx_gallery.sorting import FileNameSortKey, ExplicitOrder


# -- Project information -----------------------------------------------------

# Set project information
project = 'timescales'
copyright = '2021-{}, VoytekLab'.format(date.today().year)
author = 'Ryan Hammonds'

# Get and set the current version number
from timescales import __version__
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
    'numpydoc',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# numpydoc interacts with autosummary, that creates excessive warnings
# This line is a 'hack' for that interaction that stops the warnings
numpydoc_show_class_members = False

# Set to generate sphinx docs for class members (methods)
autodoc_default_options = {
    'members': None,
    'inherited-members': None,
}

# generate autosummary even if no references
autosummary_generate = True

# The suffix(es) of source filenames. Can be str or list of string
source_suffix = '.rst'

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
    'navbar_sidebarrel': False,
    'navbar_links': [
        ("API", "api"),
        ("Figures", 'figures'),
        #("FAQ", "faq"),
        #("Glossary", "glossary"),
        #("Motivations", "auto_motivations/index"),
        #("Tutorials", "auto_tutorials/index"),
        #("Examples", "auto_examples/index"),
        #("Reference", "reference"),
        ("GitHub", "https://github.com/voytekresearch/timescale-methods", True),
    ],

    # Set the page width to not be restricted to hardset value
    'body_max_width': None,

    # Bootswatch (http://bootswatch.com/) theme to apply.
    'bootswatch_theme': "flatly",

    # Render the current pages TOC in the navbar
    'navbar_pagenav': False,
}

# Settings for whether to copy over and show link rst source pages
html_copy_source = False
html_show_sourcelink = False


# -- Extension configuration -------------------------------------------------

# Configurations for sphinx gallery
sphinx_gallery_conf = {
    'examples_dirs': ['../figures'],
    'gallery_dirs': ['auto_figures'],
    #'subsection_order' : ExplicitOrder(['../figures/x',
    #                                    '../figures/y']),
    'within_subsection_order': FileNameSortKey,
    'backreferences_dir': 'generated',   # Where to drop linking files between examples & API
    'doc_module': ('timescales',),
    'reference_url': {'timescales': None},
    'remove_config_comments': True,
}
