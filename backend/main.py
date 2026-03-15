from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import os
import random
import uuid
from fpdf import FPDF
from io import BytesIO

from backend.database import supabase
from services.doc_chat import ask_doctor_ai
from backend.pipeline import run_segmentation

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LoginRequest(BaseModel):
    email: str
    password: str

# --- Utilities ---
def sanitize_for_pdf(text):
    if not text: return ""
    replacements = {'\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"', '\u2013': '-', '\u2014': '-', '\u2022': '*', '\u00b3': '3', '**': ''}
    for u, s in replacements.items(): text = text.replace(u, s)
    return text.encode('latin-1', 'ignore').decode('latin-1')

# --- Auth & Profile ---
class SignupRequest(BaseModel):
    name: str
    email: str
    password: str
    hospital: str

@app.post("/doctor/signup")
def doctor_signup(request: SignupRequest):
    response = supabase.auth.sign_up({"email": request.email, "password": request.password})
    user_id = response.user.id
    supabase.table("doctors").insert({"id": user_id, "name": request.name, "email": request.email, "hospital": request.hospital}).execute()
    return {"message": "Doctor created", "doctor_id": user_id}

@app.post("/doctor/login")
def doctor_login(request: LoginRequest):
    res = supabase.auth.sign_in_with_password({"email": request.email, "password": request.password})
    return {"message": "Login successful", "doctor_id": res.user.id, "session": res.session}

@app.get("/doctor/{doctor_id}")
def get_doctor(doctor_id: str):
    res = supabase.table("doctors").select("*").eq("id", doctor_id).execute()
    return res.data[0] if res.data else {"error": "Not found"}

# --- Patient Management ---
@app.post("/patients")
def create_patient(name: str, age: int, gender: str, phone: str, height: float, weight: float, doctor_id: str):
    res = supabase.table("patients").insert({"name": name, "age": age, "gender": gender, "phone": phone, "height": height, "weight": weight, "doctor_id": doctor_id}).execute()
    return {"message": "Patient created", "patient": res.data}

@app.get("/patients")
def get_all_patients(doctor_id: str):
    res = supabase.table("patients").select("*").eq("doctor_id", doctor_id).execute()
    return res.data

# --- MRI Processing ---
@app.post("/upload-mri")
async def upload_mri(patient_id: str = Form(...), file: UploadFile = File(...)):
    try:
        p_res = supabase.table("patients").select("age, gender").eq("id", patient_id).execute()
        if not p_res.data:
            raise Exception("Patient not found")

        age, gender = p_res.data[0]['age'], p_res.data[0]['gender']

        # Ensure the uploads directory is absolute and exists
        base_dir = os.path.abspath(os.path.dirname(__file__))
        uploads_dir = os.path.join(base_dir, "..", "uploads")
        os.makedirs(uploads_dir, exist_ok=True)

        file_bytes = await file.read()
        scan_id = str(uuid.uuid4())
        file_name = f"{scan_id}_{file.filename}"
        file_path = os.path.join(uploads_dir, file_name)

        # Upload scan to Supabase storage (unique filename per scan)
        supabase.storage.from_("mri-scans").upload(file_name, file_bytes)

        scan_res = supabase.table("scans").insert({
            "id": scan_id,
            "patient_id": patient_id,
            "file_name": file_name
        }).execute()

        new_scan_id = scan_id

        with open(file_path, "wb") as f:
            f.write(file_bytes)

        print("Running segmentation pipeline...")

        pipeline_result = run_segmentation(file_path, age, gender, scan_id=new_scan_id)

        print("Pipeline result:", pipeline_result)

        supabase.table("analysis").insert({
            "scan_id": new_scan_id,

            "tumor_detected": bool(pipeline_result.get("tumor_detected", False)),

            "tumor_volume": float(pipeline_result.get("WT", 0)),
            "enhancing_ratio": float(pipeline_result.get("ET", 0)),
            "core_ratio": float(pipeline_result.get("TC", 0)),
            "edema_ratio": float(pipeline_result.get("ED", 0)),

            "confidence": 0.9,
            "explanation": str(pipeline_result.get("explanation", "Analysis complete."))
        }).execute()

        return {
            "message": "Success",
            "scan_id": new_scan_id,
            "analysis": pipeline_result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/scan/slice")
def get_slice(file_name: str, slice_index: int):
    path = os.path.abspath(f"uploads/{file_name}")
    try:
        img = nib.load(path)
        data = img.get_fdata()
        slice_img = data[:, :, min(slice_index, data.shape[2]-1)]
        fig, ax = plt.subplots(); ax.imshow(slice_img, cmap="gray"); ax.axis("off")
        buf = io.BytesIO(); plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0); plt.close(fig); buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e: return {"error": str(e)}, 500

@app.get("/scan/{scan_id}/report/download")
def download_report(scan_id: str):
    try:
        analysis = supabase.table("analysis").select("*").eq("scan_id", scan_id).execute()
        patient_id = supabase.table("scans").select("patient_id").eq("id", scan_id).execute().data[0]['patient_id']
        p_data = supabase.table("patients").select("*").eq("id", patient_id).execute().data[0]
        pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "NeuroVision AI - Clinical Report", ln=True, align='C'); pdf.ln(10)
        pdf.set_font("Arial", '', 11)
        pdf.multi_cell(0, 8, sanitize_for_pdf(analysis.data[0]['explanation']))
        return Response(content=pdf.output(dest='S').encode('latin-1'), media_type="application/pdf")
    except Exception as e: return {"error": str(e)}, 500

# Complete get_patient logic for main.py
@app.get("/patient/{patient_id}")
def get_patient(patient_id: str):
    # Fetch the patient by ID
    result = supabase.table("patients").select("*").eq("id", patient_id).execute()

    # If no data is found, return a 404 error that the frontend can catch
    if not result.data or len(result.data) == 0:
        raise HTTPException(status_code=404, detail="Patient not found in database")

    # Return the first (and only) patient object
    return result.data[0]

@app.get("/patient/{patient_id}/scans")
def get_patient_scans(patient_id: str):
    # Ensure this strictly queries the scans associated with the UUID
    scans = supabase.table("scans").select("*").eq("patient_id", patient_id).execute()
    return scans.data

# main.py

@app.get("/scan/{scan_id}/analysis")
def get_analysis(scan_id: str):
    # Fetch record from the analysis table
    analysis = supabase.table("analysis").select("*").eq("scan_id", scan_id).execute()

    # If no record exists yet, return a status the frontend can read
    if not analysis.data or len(analysis.data) == 0:
        return {"status": "processing"}

    # Return the single object directly
    return analysis.data[0]

@app.get("/scan/{scan_id}/generate-image")
def generate_scan_image(scan_id: str):
    try:
        # Get scan details
        scan_res = supabase.table("scans").select("file_name, patient_id").eq("id", scan_id).execute()
        if not scan_res.data:
            raise HTTPException(status_code=404, detail="Scan not found")

        file_name = scan_res.data[0]["file_name"]
        patient_id = scan_res.data[0]["patient_id"]

        # Construct file path
        base_dir = os.path.abspath(os.path.dirname(__file__))
        uploads_dir = os.path.join(base_dir, "..", "uploads")
        file_path = os.path.join(uploads_dir, file_name)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Scan file not found")

        # Generate image
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            output_path = tmp_file.name

        run_inference(file_path, output_path=output_path)

        return FileResponse(output_path, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/ask")
def ask_ai(question: str):
    return {"answer": ask_doctor_ai(question)}