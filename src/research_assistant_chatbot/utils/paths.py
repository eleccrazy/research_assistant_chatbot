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

SRC_DIR = os.path.join(ROOT_DIR, "src")
APP_CONFIG_FPATH = os.path.join(SRC_DIR, "config", "config.yaml")
PROMPT_CONFIG_FPATH = os.path.join(SRC_DIR, "config", "prompt_config.yaml")

ENV_FPATH = os.path.join(ROOT_DIR, ".env")


DATA_DIR = os.path.join(ROOT_DIR, "data")
PUBLICATION_FPATH = os.path.join(DATA_DIR, "project_1_publications.json")
