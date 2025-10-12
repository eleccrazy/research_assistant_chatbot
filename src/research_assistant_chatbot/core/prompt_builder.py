"""
File: prompt_builder.py
Description: This module provides functions for building prompts for the llm.
Author: Gizachew Kassa
Date Created: 12/10/2025
"""
from typing import Union, List, Dict, Any

def lowercase_first_char(text: str) -> str:
    """Lowercases the first character of a string.

    Args:
        text: Input string.

    Returns:
        The input string with the first character lowercased.
    """
    return text[0].lower() + text[1:] if text else text

def format_prompt_section(lead_in: str, value: Union[str, List[str]]) -> str:
    """Formats a prompt section by joining a lead-in with content.

    Args:
        lead_in: Introduction sentence for the section.
        value: Section content, as a string or list of strings.

    Returns:
        A formatted string with the lead-in followed by the content.
    """
    if isinstance(value, list):
        formatted_value = "\n".join(f"- {item}" for item in value)
    else:
        formatted_value = value
    return f"{lead_in}\n{formatted_value}"

def build_system_prompt_from_config(config: Dict[str, Any]) -> str:
    """Builds a complete system prompt string from a configuration dictionary.

    Args:
        config: Dictionary specifying system prompt components (role, instruction, constraints, etc.).

    Returns:
        A formatted system prompt string ready for use in an LLM system message.

    Raises:
        ValueError: If the required 'role' field is missing.
    """
    prompt_parts = []

    # Role: define assistant's identity
    role = config.get("role")
    if not role:
        raise ValueError("Missing required 'role' field.")
    prompt_parts.append(f"You are {lowercase_first_char(role.strip())}")

    # Instruction
    if instruction := config.get("instruction"):
        prompt_parts.append(format_prompt_section("Your Instruction:", instruction))

    # Constraints / guidelines
    if constraints := config.get("output_constraints"):
        prompt_parts.append(format_prompt_section("Follow these important guidelines:", constraints))

    # Communication style
    if tone := config.get("style_or_tone"):
        prompt_parts.append(format_prompt_section("Communication style:", tone))

    # Output format
    if output_format := config.get("output_format"):
        prompt_parts.append(format_prompt_section("Response formatting:", output_format))

    return "\n\n".join(prompt_parts)
