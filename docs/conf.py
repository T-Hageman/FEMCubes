# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from importlib.metadata import PackageNotFoundError, version as _installed_version

from pybtex.plugin import register_plugin
from pybtex.style.formatting.unsrt import Style as UnsrtStyle
from pybtex.style.labels import BaseLabelStyle

# -- Path setup ---------------------------------------------------------------
# Add the source folder so autodoc can import femcubes
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information ------------------------------------------------------
project = "FEMCubes"
copyright = "2026, FEMCubes contributors"
author = "FEMCubes contributors"
try:
    # Set from the git tag when FEMCubes is installed
    release = _installed_version("femcubes")
except PackageNotFoundError:
    release = "unknown"
version = release

# -- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",           # Pull docstrings from code
    "sphinx.ext.autosummary",       # Generate summary tables
    "sphinx.ext.napoleon",          # Support NumPy/Google docstrings
    "sphinx.ext.intersphinx",       # Link to external docs (NumPy, etc.)
    "sphinx.ext.viewcode",          # Add [source] links to docs
    "sphinx.ext.mathjax",           # Render LaTeX math
    "sphinxcontrib.bibtex",         # Citations from references.bib
]

# Napoleon settings (NumPy-style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
# Class attributes listed under "Attributes" are also picked up by autodoc
# (undoc-members); rendering the section as fields avoids duplicate entries.
napoleon_use_ivar = True
# Section headers used in the FEMCubes docstrings next to the NumPy ones
napoleon_custom_sections = [
    "Public API",
    "Members",
    "Example model settings",
    "Required parameters",
    "Optional parameters",
    "Optional parameters (continued)",
    "Named markers",
    "Boundary groups",
    "Supported dimensions",
    "Local operator split",
    "Local cohesive problem",
    "Elastic law and phase-field degradation",
    "Elastic constitutive law",
    "Consistent tangent",
    "Mesh-size correction of the fracture toughness (optional)",
]

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Autosummary
autosummary_generate = True

# Intersphinx: link to external project docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# BibTeX citations (sphinxcontrib-bibtex). Pages and docstrings cite with
# :cite:p:`key` or :cite:t:`key`; each page that cites lists its own
# references with
#
#   .. bibliography::
#      :filter: docname in docnames
#
# A reference cited on several pages is then listed on each of them. The
# citations of a page link to the list on that page, so the duplicate-citation
# warning is suppressed. The entries are labelled by the surname of their first
# author and their year (Bourdin2000), so that the labels match the
# author-year citations in the text and stay the same on every page (the
# "alpha" labels are formed per page and would clash). Papers with the same
# authors and year carry a letter in their year field in references.bib.
class _BibAuthorYearLabelStyle(BaseLabelStyle):
    """Label each reference by its first author's surname and its year."""

    def format_labels(self, sorted_entries):
        for entry in sorted_entries:
            persons = entry.persons.get("author") or entry.persons.get("editor")
            if persons:
                name = "".join(persons[0].prelast_names + persons[0].last_names)
            else:
                name = entry.key
            label = name + entry.fields.get("year", "")
            yield "".join(char for char in label if char.isalnum())


class _BibStyle(UnsrtStyle):
    """Entry formatting of pybtex's "unsrt" style, with author-year labels, sorted by author, year and title."""

    default_label_style = "femcubes_author_year"
    default_sorting_style = "author_year_title"


register_plugin("pybtex.style.labels", "femcubes_author_year", _BibAuthorYearLabelStyle)
register_plugin("pybtex.style.formatting", "femcubes", _BibStyle)

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "femcubes"
bibtex_reference_style = "author_year"
suppress_warnings = ["bibtex.duplicate_citation"]

# -- Options for HTML output --------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/T-Hageman/FEMCubes",
    "show_toc_level": 2,
    "logo": {
        "image_light": "_static/logo/femcubes_logo.svg",
        "image_dark": "_static/logo/femcubes_logo_dark.svg",
        "alt_text": "FEMCubes",
    },
}
html_static_path = ["_static"]
html_favicon = "_static/logo/favicon.png"

# Suppress warnings for missing references to external FEniCSx types
nitpicky = False

# Mock imports that may not be available at doc-build time.
# All dependencies are present in the FenicsX conda environment,
# so no mocking is needed.  Uncomment if building docs outside this env.
# autodoc_mock_imports = [
#     "dolfinx",
#     "ufl",
#     "basix",
#     "petsc4py",
#     "mpi4py",
#     "pyvista",
#     "gmsh",
#     "adios4py",
# ]
