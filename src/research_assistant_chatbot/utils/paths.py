"""
File: paths.py
Description: This module defines the file paths used in the project.
Author: Gizachew Kassa
Date: 21/09/2025
"""

import os

ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
)

ENV_FPATH = os.path.join(ROOT_DIR, ".env")


DATA_DIR = os.path.join(ROOT_DIR, "data")
PUBLICATION_FPATH = os.path.join(DATA_DIR, "project_1_publications.json")

print(ROOT_DIR)
