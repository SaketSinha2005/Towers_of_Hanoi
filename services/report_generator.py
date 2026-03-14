from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from backend.config import OPENAI_API_KEY

llm = ChatOpenAI(
api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
    temperature=0.2
)

report_prompt = PromptTemplate(
    input_variables=[
        "age",
        "gender",
        "edema",
        "core",
        "enhancing",
        "volume"
    ],
    template="""
Generate a structured MRI tumor analysis report.

Patient:
Gender: {gender}
Age: {age}

Tumor metrics:
Edema ratio: {edema}
Tumor core ratio: {core}
Enhancing tumor ratio: {enhancing}
Total tumor volume: {volume} cm³

The report should contain these sections:

MRI ANALYSIS REPORT
Tumor Segmentation Findings
Interpretation
Precautions

Do NOT recommend medications.
Only suggest monitoring or lifestyle precautions.
"""
)

def generate_report(age, gender, edema, core, enhancing, volume):

    prompt = report_prompt.format(
        age=age,
        gender=gender,
        edema=edema,
        core=core,
        enhancing=enhancing,
        volume=volume
    )

    response = llm.invoke(prompt)

    return response.content
