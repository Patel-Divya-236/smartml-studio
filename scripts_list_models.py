"""List the models your configured provider currently serves.

    python scripts_list_models.py

Reads LLM_API_KEY / LLM_BASE_URL from the environment or your secrets file.
Use this instead of trusting a hardcoded model id — providers retire ids regularly.
"""

import sys

from backend.core.secrets import load_llm_env
from src.llm.client import LLMClient

load_llm_env()

client = LLMClient()

if not client.is_available:
    sys.exit(
        "No LLM_API_KEY found.\n"
        "  cp secrets.toml.example secrets.toml\n"
        "  then add your provider key to it."
    )

print(f"Provider : {client.base_url}")
print(f"Configured model in settings: {client.model}\n")

models = client.list_models()
if models is None:
    sys.exit(f"Could not list models: {client.last_error}")

print(f"{len(models)} models available:\n")
for name in models:
    marker = "  <-- currently configured" if name == client.model else ""
    print(f"  {name}{marker}")

if client.model not in models:
    print(
        f"\n!  '{client.model}' is NOT in this provider's list — requests will 404.\n"
        f"   Set LLM_MODEL in secrets.toml to one of the above."
    )

print(
    "\nWhat to pick: an *instruct* model (not base), the largest the free tier offers.\n"
    "Size matters most for the grounded-explanation tasks, where the model must stick\n"
    "to the numbers it was given instead of inventing plausible ones."
)
