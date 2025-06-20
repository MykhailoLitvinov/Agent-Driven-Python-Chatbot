# 🧠 Agent-Driven Python Chatbot

An intelligent, modular chatbot system built with Python that features automatic agent selection, LLM-powered responses, and sophisticated context management with conversation summarization.

---

## 📌 Features

- 🤖 **Three Specialized AI Agents**:
  - `@Sentinel`: Expert cybersecurity advisor for practical security solutions
  - `@FinGuide`: Personal financial advisor for budgeting and investment guidance  
  - `@EduBot`: AI learning tutor specializing in clear explanations and personalized teaching

- 🎯 **Intelligent Agent Selection**: LLM-powered automatic routing to the most appropriate agent
- 🔄 **Manual Agent Switching**: Direct agent calls using @AgentName syntax
- 🧠 **Advanced Context Management**: Maintains conversation history with intelligent summarization
- 🪄 **OpenAI Integration** with JSON schema responses for reliable outputs
- 🧩 **Clean Architecture** with configuration-driven design
- 🧾 **Comprehensive Logging** with performance metrics and conversation tracking
- ⚙️ **Fully Configurable** agents and system components via YAML files

---

## 🗂️ Project Structure

```
/config
  /interactive          # User-facing agent configurations
    sentinel.yaml       # Cybersecurity expert configuration
    finguide.yaml       # Financial advisor configuration
    edubot.yaml         # Educational tutor configuration
  /system              # Internal system configurations
    agent_selector.yaml # LLM-based agent selection configuration
    summarizer.yaml     # Conversation summarization configuration

/src
  __init__.py           # Package initialization
  agents.py             # AgentManager: LLM-based selection and agent management
  chatbot.py            # Core chatbot: input handling, context, response generation
  llm_client.py         # OpenAI API wrapper with JSON schema support
  logger.py             # Logging utilities and configuration
  memory.py             # Conversation memory and summary buffer management
  summarizer.py         # LLM-powered conversation summarization

/tests
  conftest.py           # Shared test fixtures and configurations
  test_agents.py        # Agent management and selection tests
  test_chatbot.py       # Core chatbot functionality tests
  test_llm_client.py    # LLM client and API integration tests
  test_logger.py        # Logging system tests
  test_memory.py        # Memory and context management tests
  test_summarizer.py    # Summarization functionality tests

/logs                   # Generated conversation logs
main.py                 # Console application entry point
README.md
requirements.txt        # Core dependencies
test_requirements.txt   # Testing dependencies
```

---

## 🚀 Getting Started

### 1. Clone & Install
```bash
git clone https://github.com/your-username/agent-chatbot.git
cd agent-chatbot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Setup `.env`
Create a `.env` file based on `.env.example`:
```
OPENAI_API_KEY=YOUR_API_KEY
```

### 3. Run the Chatbot
```bash
python main.py
```

---

## 💬 Sample Interactions

```bash
> How do I secure my email account?
🤖 (Auto-selected Sentinel): 
**Immediate Action**: Enable two-factor authentication (2FA) on your email account right now.

**Simple Explanation**: 2FA adds a second layer of security that makes it nearly impossible for hackers to access your account, even if they have your password...

> I want to save $500 per month
🤖 (Auto-selected FinGuide):
**Quick Assessment**: Great goal! $500/month is $6,000 annually - a solid foundation for financial security.

**Immediate Action**: Open a separate high-yield savings account today to keep this money separate from spending funds...

> @EduBot explain how neural networks work
🤖 (Directly called EduBot):
**Simple Summary**: Neural networks are computer systems that learn patterns by mimicking how brain neurons connect and communicate.

**Build Understanding**: Think of it like teaching a child to recognize cats in photos. You show thousands of cat pictures...
```

---

## ⚙️ Configuration System

The chatbot uses a sophisticated YAML-based configuration system with two types of configs:

### Interactive Agent Configs (`/config/interactive/`)
Each user-facing agent has its own configuration:

```yaml
name: FinGuide
alias: "@FinGuide"
description: Personal financial advisor focused on practical budgeting and investment guidance
temperature: 0.3
model: gpt-4o-mini
max_tokens: 1000
system_prompt: |
  You are FinGuide, a financial advisor who helps people make smart money decisions.
  
  ## Your Role
  Provide practical, easy-to-follow financial advice...
```

### System Configs (`/config/system/`)
Internal system components are also configurable:

```yaml
# agent_selector.yaml - Controls automatic agent selection
name: AgentSelector
model: gpt-4o-mini
temperature: 0.1
max_tokens: 50
system_prompt: |
  You are an expert at matching user queries to the most appropriate AI assistant...

# summarizer.yaml - Controls conversation summarization  
name: Summarizer
model: gpt-4o-mini
temperature: 0.2
max_tokens: 300
system_prompt: |
  You are an expert conversation summarizer who creates concise, accurate summaries...
```

---

## 🧪 Testing

The project includes comprehensive test coverage with 116+ unit tests:

```bash
# Install testing dependencies
pip install -r test_requirements.txt

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test modules
pytest tests/test_agents.py -v
pytest tests/test_llm_client.py -v
pytest tests/test_summarizer.py -v
```

### Test Categories
- **Agent Management**: Tests for LLM-based agent selection and configuration loading
- **LLM Integration**: Tests for OpenAI API calls with JSON schema responses
- **Memory Management**: Tests for conversation context and summarization
- **Core Functionality**: Tests for chatbot workflows and error handling
- **Logging System**: Tests for conversation logging and metrics

---

## 🏗️ Architecture Highlights

### LLM-Powered Agent Selection
- Uses OpenAI's structured output with JSON schema for reliable agent routing
- Fallback to keyword-based selection if LLM selection fails
- Configurable selection criteria and prompts

### Intelligent Context Management
- Maintains recent conversation history with configurable memory limits
- Automatic conversation summarization when memory limit is reached
- JSON-structured summaries for consistent format

### Configuration-Driven Design
- All agents and system components use YAML configuration files
- Hot-swappable prompts and parameters without code changes
- Separation of interactive (user-facing) and system configurations

### Robust Error Handling
- Graceful fallbacks for LLM failures
- Comprehensive logging with performance metrics
- Exception handling with user-friendly error messages

---

## 📈 Metrics & Logging

All conversations, performance metrics, and system events are logged in the `logs/` directory:

- **Conversation Logs**: Complete chat history with timestamps
- **Performance Metrics**: Response times, agent selection decisions, memory usage
- **Error Tracking**: LLM failures, configuration issues, system errors
- **Agent Analytics**: Usage patterns, selection accuracy, response quality

Log files are rotated daily and include structured data for analysis.

---

## 🚀 Advanced Usage

### Environment Variables
```bash
OPENAI_API_KEY=your_api_key_here
MAX_MEMORY_MESSAGES=10  # Override default memory limit
```

### Custom Agent Development
1. Create a new YAML config in `/config/interactive/`
2. Define the agent's expertise, prompts, and parameters
3. Add the agent name to the `AgentName` enum in `src/agents.py`
4. The system will automatically load and use the new agent

### Configuration Customization
- Modify agent personalities by editing system prompts
- Adjust response styles by changing temperature and max_tokens
- Customize agent selection behavior via `agent_selector.yaml`
- Fine-tune summarization with `summarizer.yaml` parameters