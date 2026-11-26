"""Central configuration: loads .env via python-dotenv, exposes settings and get_llm(role).

get_llm(role) maps an agent role to a model: Groq (fast/cheap) for supervisor routing and
guardrail checks; OpenRouter via langchain-openai (stronger) for research, risk and report.
"""
