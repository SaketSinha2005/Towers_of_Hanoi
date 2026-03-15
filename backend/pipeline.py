# backend/pipeline.py

import os

from backend.inference import run_inference
from services.report_generator import generate_report


def run_segmentation(file_path, patient_age, patient_gender, scan_id=None):

    ratios = run_inference(file_path)

    result = {
        "tumor_detected": bool(ratios["Total Tumor"] > 0.1),
        "WT": float(ratios["Total Tumor"]),      # whole tumor ratio
        "ET": float(ratios["Enhancing"]),        # enhancing tumor
        "TC": float(ratios["Necrotic"]),         # tumor core
        "ED": float(ratios["Edema"]),            # edema
    }

    report_text = generate_report(
        age=patient_age,
        gender=patient_gender,
        edema=float(result["ED"]),
        core=float(result["TC"]),
        enhancing=float(result["ET"]),
        volume=float(result["WT"])
    )

    result["explanation"] = report_text

    return result