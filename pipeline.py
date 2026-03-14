from ml.llm_explainer import explain_analysis
from ml.preprocess import load_mri

def run_segmentation(file_path):

    # load MRI
    volume = load_mri(file_path)

    # placeholder until UNet is ready
    result = {
        "tumor_detected": True,
        "ET": 0.3,
        "WT": 0.1,
        "TC": 0.2
    }

    # generate LLM explanation
    explanation = explain_analysis(result)

    result["explanation"] = explanation

    return result
