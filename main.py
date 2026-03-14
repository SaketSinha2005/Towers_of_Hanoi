from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import nibabel as nib
import matplotlib.pyplot as plt
import io

from database import supabase
from ml.pipeline import run_segmentation
from ml.llm_explainer import ask_doctor_ai

app = FastAPI()


# --------------------------------
# Home
# --------------------------------
@app.get("/")
def home():
    return {"message": "Brain Tumor AI Backend Running"}


# --------------------------------
# Doctor Signup
# --------------------------------
@app.post("/doctor/signup")
def doctor_signup(name: str, email: str, password: str, hospital: str):

    response = supabase.auth.sign_up({
        "email": email,
        "password": password
    })

    user_id = response.user.id

    supabase.table("doctors").insert({
        "id": user_id,
        "name": name,
        "email": email,
        "hospital": hospital
    }).execute()

    return {
        "message": "Doctor account created",
        "doctor_id": user_id
    }


# --------------------------------
# Doctor Login
# --------------------------------
@app.post("/doctor/login")
def doctor_login(email: str, password: str):

    response = supabase.auth.sign_in_with_password({
        "email": email,
        "password": password
    })

    return {
        "message": "Login successful",
        "session": response.session
    }


# --------------------------------
# Create Patient (linked to doctor)
# --------------------------------
@app.post("/patients")
def create_patient(
    name: str,
    age: int,
    gender: str,
    phone: str,
    height: float,
    weight: float,
    doctor_id: str
):

    result = supabase.table("patients").insert({
        "name": name,
        "age": age,
        "gender": gender,
        "phone": phone,
        "height": height,
        "weight": weight,
        "doctor_id": doctor_id
    }).execute()

    return {
        "message": "Patient created successfully",
        "patient": result.data
    }
# --------------------------------
# Upload MRI and run analysis
# --------------------------------
@app.post("/upload-mri")
async def upload_mri(patient_id: str, file: UploadFile = File(...)):

    file_bytes = await file.read()
    file_name = file.filename

    # upload MRI to storage
    supabase.storage.from_("mri-scans").upload(file_name, file_bytes)

    # save locally for ML processing
    file_path = f"uploads/{file_name}"

    with open(file_path, "wb") as f:
        f.write(file_bytes)

    # create scan record
    scan = supabase.table("scans").insert({
        "patient_id": patient_id,
        "file_name": file_name
    }).execute()

    scan_id = scan.data[0]["id"]

    # run segmentation pipeline
    result = run_segmentation(file_path)

    # store analysis results including LLM explanation
    supabase.table("analysis").insert({
        "scan_id": scan_id,
        "tumor_detected": result["tumor_detected"],
        "tumor_volume": result["WT"],
        "confidence": 0.9,
        "explanation": result["explanation"]
    }).execute()

    return {
        "message": "MRI uploaded and analyzed",
        "analysis": result
    }


# --------------------------------
# Get scans for a patient
# --------------------------------
@app.get("/patient/{patient_id}/scans")
def get_patient_scans(patient_id: str):

    scans = supabase.table("scans") \
        .select("*") \
        .eq("patient_id", patient_id) \
        .execute()

    return scans.data


# --------------------------------
# Get analysis results
# --------------------------------
@app.get("/scan/{scan_id}/analysis")
def get_analysis(scan_id: str):

    analysis = supabase.table("analysis") \
        .select("*") \
        .eq("scan_id", scan_id) \
        .execute()

    return analysis.data


# --------------------------------
# MRI Slice Viewer
# --------------------------------
@app.get("/scan/slice")
def get_slice(file_name: str, slice_index: int):

    file_path = f"uploads/{file_name}"

    img = nib.load(file_path)
    data = img.get_fdata()

    slice_img = data[:, :, slice_index]

    fig, ax = plt.subplots()
    ax.imshow(slice_img, cmap="gray")
    ax.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


# --------------------------------
# AI Doctor Question Endpoint
# --------------------------------
@app.post("/ai/ask")
def ask_ai(question: str):

    answer = ask_doctor_ai(question)

    return {
        "answer": answer
    }
