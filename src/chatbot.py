from datetime import datetime

from agents import AgentManager
from llm_client import LLMClient
from memory import ConversationMemory
from utils import setup_logging


class Chatbot:
    """Main chatbot class with agent support"""

    def __init__(self):
        self.llm_client = LLMClient()
        self.memory = ConversationMemory()
        self.default_agent = self.current_agent = "EduBot"  # TODO: validate agent name
        self.agent_manager = AgentManager(self.llm_client, self.default_agent)
        self.logger = setup_logging()

    def start(self):
        """Run the console interface"""
        while True:
            try:
                user_input = input(f"\n👤 You: ").strip()

                if not user_input:
                    continue

                # Handle system commands
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == "reset":
                    self.reset_conversation()
                    continue
                elif user_input.lower() == "help":
                    self.show_help()
                    continue

                # Handle regular user query
                response = self.process_query(user_input)
                print(f"🤖 {self.current_agent}: {response}")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                self.logger.error(f"Error in main loop: {str(e)}")

    def process_query(self, query: str) -> str:
        """Process the user's query"""
        start_time = datetime.now()

        # Log user message
        self.logger.info(f"Retrieved user message: {query}")

        # Select the appropriate agent
        selected_agent = self.agent_manager.select_agent(query)

        # Notify if agent is switched
        if selected_agent != self.current_agent:
            print(f"🔄 Switching to {selected_agent}")
            self.current_agent = selected_agent

        # Add user query to memory
        self.memory.add_message("user", query)

        # Generate response
        context = self.memory.get_context()
        response = self.agent_manager.generate_response(selected_agent, context)

        # Log agent response
        self.logger.info(f"Retrieved response from {selected_agent}: {response}")

        # Save response to memory
        self.memory.add_message("assistant", response, self.current_agent)

        # Log processing time
        duration = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Query processed by {selected_agent} in {duration:.2f}s")

        return response

    def reset_conversation(self):
        """Reset the entire conversation"""
        self.memory.reset()
        self.current_agent = self.default_agent
        print("🔄 Conversation reset!")
        self.logger.info("Conversation reset")

    @staticmethod
    def show_help():
        """Display help information"""
        help_text = """
🤖 AI Chatbot Help:

AGENTS:
  @Sentinel  - Cybersecurity advisor
  @FinGuide  - Financial consultant  
  @EduBot    - Educational tutor

COMMANDS:
  reset      - Reset conversation
  quit       - Exit chatbot
  help       - Show this help

USAGE:
  - Ask any question naturally
  - Use @AgentName to directly call an agent
  - System will auto-select best agent otherwise
        """
        print(help_text)
