"""
File: cli.py
Description: Command-line interface for interactive chatting with the RAG research assistant chatbot.
Author: Gizachew Kassa
Date Created: 25/10/2025
"""

import sys
import time

def run_cli(chatbot):
    """Runs an interactive CLI chat session using a pre-initialized chatbot."""
    print("\nResearch Assistant Chatbot (CLI Mode)")
    print("Type 'exit' or 'quit' to end the session.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("\nExiting chat. Goodbye!\n")
                break

            if not user_input:
                continue

            response = chatbot.ask(user_input)
            print(f"\nAssistant: {response.get('response', '').strip()}\n")
            time.sleep(1)

        except KeyboardInterrupt:
            print("\n\nChat session interrupted. Exiting gracefully...\n")
            sys.exit(0)
        except Exception as e:
            print(f"\nError: {e}\n")
