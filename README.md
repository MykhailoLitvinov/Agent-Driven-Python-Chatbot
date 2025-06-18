# 🧠 Agent-Driven Chatbot (Technical Test – Softcery)

A modular, console-based chatbot built with Python that supports agent switching, LLM integration, and dynamic context management using summary buffers.

---

## 📌 Features

- 🤖 **Multiple Specialized Agents**:
  - `@Sentinel`: Cybersecurity Advisor
  - `@FinGuide`: Financial Consultant
  - `@EduBot`: AI-Powered Tutor

- 🔄 **Automatic or Manual Agent Switching**
- 🧠 **Context Management**: Keeps last `N` messages + summarized history
- 🪄 **LLM Integration** via OpenAI (pluggable)
- 🧩 **Clean Architecture** following SOLID principles
- 🧾 **Conversation Logging** + Performance Metrics
- ⚙️ **Configurable Agents** via YAML

---

## 🗂️ Project Structure

```
/config
    sentinel.yaml       # Agent config: prompt, keywords, temperature
    finguide.yaml
    edubot.yaml

/src
    agents.py           # AgentManager: selects and manages agents
    chatbot.py          # Chatbot core: handles input, context, response
    llm_client.py       # LLM wrapper (e.g., OpenAI)
    memory.py           # Context & summary buffer
    utils.py            # Helpers: config loading, logging, formatting

/tests
    ...                 # Unit tests (pytest)

.env                    # Contains API key, config vars
main.py                 # Console entry point
README.md
requirements.txt
test_requirements.txt
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
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-3.5-turbo
SUMMARY_MESSAGE_LIMIT=5
```

### 3. Run the Chatbot
```bash
python main.py
```

---

## 💬 Sample Interaction

```bash
> How do I secure my email account?
(Sentinel): Enable two-factor authentication, use strong passwords...

> How can I budget for security software?
(FinGuide): Set aside a monthly fixed amount...

> I want to understand encryption methods.
(EduBot): Encryption transforms data into unreadable formats...
```

---

## ⚙️ Agent Configuration (YAML)

Each agent is fully configurable via a YAML file:

```yaml
name: FinGuide
alias: "@FinGuide"
temperature: 0.4
keywords: [money, budget, cost, ...]
system_prompt: |
  You are a certified financial advisor...
```

See files in the `/config` directory.

---

## 🧪 Testing

```bash
pip install -r test_requirements.txt
pytest tests/
```

---

## 📈 Metrics & Logging

All conversations and performance metrics (response time, agent used) are logged in `logs/`.

---

## 🔐 Repository Access

Once complete, share the **private GitHub repository** with:

```
GitHub user: maisterr
```