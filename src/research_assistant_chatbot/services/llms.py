"""
File: llms.py
Description: This module provides functions to get language model instances based on specified models.
Author: Gizachew Kassa
Date Created: 22/09/2025
"""

import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from utils.helpers import load_env

from typing import Union


available_models = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "llama3-8b-8192",
]


def get_llm(model: str) -> Union["ChatGoogleGenerativeAI", "ChatGroq"]:
    """
        Returns an instance of a chat language model based on the specified model name.

        This function loads environment variables and validates the requested model
        against the available models. Depending on the model, it initializes and
        returns the appropriate LLM client with the corresponding API key and settings.

        Parameters:
            model (str): The name of the language model to initialize. Must be one of
                         the keys in `available_models`.

        Returns:
            Union[ChatGoogleGenerativeAI, ChatGroq]: An instance of the requested chat model
                                                     client, ready to use.

        Raises:
            ValueError: If the provided model name is not in `available_models`.
        """
    load_env()

    if model not in available_models:
        raise ValueError(f"Invalid model. Available models: {available_models.keys()}")

    if model in ["gemini-1.5-flash", "gemini-1.5-pro"]:
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=0.0,
            api_key=os.getenv("GOOGLE_API_KEY"),
        )

    elif model == "llama3-8b-8192":
        return ChatGroq(
            model_name="llama3-8b-8192",
            temperature=0.0,
            api_key=os.getenv("GROQ_API_KEY"),
        )