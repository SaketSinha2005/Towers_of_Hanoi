from langchain_openai import ChatOpenAI
from backend.config import OPENAI_API_KEY
import os

os.environ["LANGCHAIN_TRACING_V2"] = "false"

llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
    temperature=0.3
)

SYSTEM_PROMPT = """
You are an AI assistant helping doctors interpret brain tumor MRI segmentation results.

Safety rules:

1. NEVER recommend medications.
2. NEVER suggest alternative drugs.
3. NEVER give dosage instructions.
4. If asked about a medication:
   - Only explain its medical purpose or function.
5. You may discuss:
   - tumor interpretation
   - MRI findings
   - monitoring recommendations
   - general medical information
6. Always remind that treatment decisions must be made by qualified physicians.
"""

def ask_doctor_ai(question):

    prompt = SYSTEM_PROMPT + "\nDoctor question: " + question

    response = llm.invoke(prompt)

    return response.content

