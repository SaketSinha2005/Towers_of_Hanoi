import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { UploadCloud } from 'lucide-react';

export default function UploadMRI() {
  const navigate = useNavigate();
  const [patientId, setPatientId] = useState('');
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);

  // Inside handleProcessScan in UploadMRI.jsx
const handleProcessScan = async (e) => {
  e.preventDefault();
  if (!file || !patientId) return alert("Patient ID and .nii file are required.");
  setIsUploading(true);

  try {
    const formData = new FormData();
    formData.append('file', file);
    // FIX: Add the patient_id to the Form Data body
    formData.append('patient_id', patientId);

    const res = await fetch(`http://127.0.0.1:8000/upload-mri`, {
      method: 'POST',
      body: formData
      // No headers needed; fetch sets multipart/form-data automatically
    });

    if (!res.ok) {
      const errorData = await res.json();
      throw new Error(errorData.detail || "Upload failed");
    }

    navigate(`/dashboard/patient/${patientId}`);
  } catch (err) {
    alert(err.message);
  } finally {
    setIsUploading(false);
  }
};
  return (
    <div style={{ maxWidth: '600px', margin: '4rem auto' }}>
      <div className="card" style={{ padding: '3rem', textAlign: 'center' }}>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', marginBottom: '2rem' }}>Upload MRI Scan (.nii)</h2>
        <form onSubmit={handleProcessScan}>
          <div className="form-group" style={{ textAlign: 'left', marginBottom: '1.5rem' }}>
            <label className="form-label">Patient UUID</label>
            <input type="text" className="form-input" value={patientId} onChange={(e) => setPatientId(e.target.value)} required />
          </div>
          <div style={{ border: '2px dashed #93c5fd', borderRadius: '12px', padding: '4rem', backgroundColor: '#eff6ff', position: 'relative', marginBottom: '2rem' }}>
            <input type="file" style={{ position: 'absolute', inset: 0, opacity: 0, cursor: 'pointer' }} onChange={(e) => setFile(e.target.files[0])} accept=".nii,.nii.gz" />
            <UploadCloud size={48} color="#2563eb" />
            <p style={{ marginTop: '1rem', fontWeight: '500' }}>{file ? file.name : 'Click or Drag NIfTI File'}</p>
          </div>
          <button type="submit" className="btn-primary" style={{ width: '100%', height: '3.5rem' }} disabled={isUploading}>
            {isUploading ? 'Processing Pipeline...' : 'Run Analysis'}
          </button>
        </form>
      </div>
    </div>
  );
}