import os
import sys

sys.path.insert(0, os.path.abspath('..'))
project = 'MediMind'
copyright = '2024, Shivam Goyal'
author = 'Shivam Goyal'
release = '0.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_show_sphinx = False
html_title = 'MediMind Documentation'
html_context = {
    'display_github': True,
    'github_user': 'ShivamGoyal03',
    'github_repo': 'MediMind',
    'github_version': 'main',
    'commit': False,
}