from src.chatbot import Chatbot
from src.agents import AgentName


def main():
    """Main function to start the chatbot"""
    print("🤖 AI Agent Chatbot v1.0")
    print("=" * 50)
    print(f"Available agents: {AgentName.values()}")
    print("Commands: 'reset', 'quit', 'exit'")
    print("=" * 50)

    chatbot = Chatbot()
    chatbot.start()


if __name__ == "__main__":
    main()
