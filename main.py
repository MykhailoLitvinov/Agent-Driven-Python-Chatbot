from src.chatbot import Chatbot


def main():
    """Main function to start the chatbot"""
    print("🤖 AI Agent Chatbot v1.0")
    print("=" * 50)
    print("Available agents: @Sentinel, @FinGuide, @EduBot")
    print("Commands: 'reset', 'quit', 'help'")
    print("=" * 50)

    chatbot = Chatbot()
    chatbot.start()


if __name__ == "__main__":
    main()
