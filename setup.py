"""
Setup script for BRIDGE: Biological Regulatory Integration for Dynamic Gene Expression.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

# Read requirements
requirements = []
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="bridge-omics",
    version="1.0.0",
    author="Manus AI",
    author_email="contact@manus.ai",
    description="BRIDGE: Biological Regulatory Integration for Dynamic Gene Expression - A framework for integrating SCENIC+ and PINNACLE for multi-omic network analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/manus-ai/bridge",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
        "viz": [
            "plotly>=5.0",
            "umap-learn>=0.5",
        ],
        "all": [
            "pytest>=6.0",
            "pytest-cov>=2.0", 
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "plotly>=5.0",
            "umap-learn>=0.5",
        ]
    },
    entry_points={
        "console_scripts": [
            "bridge=bridge.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "bridge": ["data/*.json", "data/*.txt"],
    },
    keywords="bioinformatics, single-cell, gene-regulation, protein-networks, scenic, pinnacle, multi-omics, bridge",
    project_urls={
        "Bug Reports": "https://github.com/manus-ai/bridge/issues",
        "Source": "https://github.com/manus-ai/bridge",
        "Documentation": "https://bridge-omics.readthedocs.io/",
    },
)

