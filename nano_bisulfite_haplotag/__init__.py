"""
Nano Bisulfite Haplotag - A tool for haplotype tagging of nanopore bisulfite sequencing data
"""

__version__ = "1.0.0"
__author__ = "Xiaohui Xue"
__email__ = "xxhui@alumni.pku.edu.cn"

from .haplotag import HaploTagger
from .main import main

__all__ = ["HaploTagger", "main"]