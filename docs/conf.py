import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "NeuroBench"
copyright = "2024, Jason Yik, Noah Pacik-Nelson, Korneel Van Den Berghe"
author = "Jason Yik, Noah Pacik-Nelson, Korneel Van Den Berghe"
release = "1.0.6"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    # 'sphinx.ext.intersphinx'
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

master_doc = "index"

autodoc_mock_imports = ["torch", "torchaudio", "metavision_ml", "pytorch_lightning"]
todo_include_todos = True

autodoc_default_options = {
    "special-members": "__call__, __init__",  # Include __call__ in the docs
    "undoc-members": True,  # Include undocumented members
    "show-inheritance": True,  # Show inheritance hierarchy
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"

html_static_path = ["_static"]

pygments_style = "sphinx"
