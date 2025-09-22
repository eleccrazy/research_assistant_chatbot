"""
helpers.py

This module provides utility functions for the project, including:

- Loading publications from a JSON or Markdown file.
- Loading and parsing YAML configuration files.
- Loading environment variables from a .env file and verifying required keys.

Author: Gizachew Kassa
Date: 22/09/2025
"""

import os
from pathlib import Path
from typing import Union

import yaml
from dotenv import load_dotenv
from utils.paths import ENV_FPATH, PUBLICATION_FPATH


def load_publications() -> None:
    """Loads list of publications from a markdown file.

    Returns:
        Content of the publication as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there's an error reading the file.
    """
    file_path = Path(PUBLICATION_FPATH)

    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"Publication file not found: {file_path}")

    # Read and return the file content
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except IOError as e:
        raise IOError(f"Error reading publication file: {e}") from e


def load_yaml_config(file_path: Union[str, Path]) -> dict:
    """Loads a YAML configuration file.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If there's an error parsing YAML.
        IOError: If there's an error reading the file.
    """
    file_path = Path(file_path)

    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"YAML config file not found: {file_path}")

    # Read and parse the YAML file
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}") from e
    except IOError as e:
        raise IOError(f"Error reading YAML file: {e}") from e


def load_env() -> None:
    """Loads environment variables from a .env file and checks for required keys.

    Raises:
        AssertionError: If required keys are missing.
    """
    # Load environment variables from .env file
    load_dotenv(ENV_FPATH, override=True)

    # Check if 'XYZ' has been loaded
    api_key = os.getenv("GOOGLE_API_KEY")

    assert api_key, "'api_key' has not been loaded or is not set in the .env file."
