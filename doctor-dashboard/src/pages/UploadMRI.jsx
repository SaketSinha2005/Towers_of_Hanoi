import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { UploadCloud, File, CheckCircle } from 'lucide-react';
import { dataStore } from '../lib/dummyData';

export default function UploadMRI() {
  const navigate = useNavigate();
  const [patientId, setPatientId] = useState('');
  const [files, setFiles] = useState([]);
  const [isUploading, setIsUploading] = useState(false);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setFiles(Array.from(e.target.files));
    }
  };

  const handleProcessScan = (e) => {
    e.preventDefault();
    if (!patientId || files.length === 0) {
      alert("Please provide a Patient ID and select at least one file.");
      return;
    }

    const patient = dataStore.getPatient(patientId);
    if(!patient) {
        alert("Patient ID not found. Please create the patient first.");
        return;
    }

    setIsUploading(true);
    
    // Simulate upload delay
    setTimeout(() => {
      // Mock analyzing text and returning
      const filenames = files.map(f => f.name);
      dataStore.addCheckup(patientId, filenames);
      
      setIsUploading(false);
      navigate(`/dashboard/analysis/${patientId}`);
    }, 1500);
  };

  return (
    <div style={{ maxWidth: '500px', margin: '0 auto', display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 'calc(100vh - 4rem)' }}>
      <div className="card" style={{ width: '100%', padding: '2.5rem', border: '1px solid #e2e8f0', boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)', textAlign: 'center' }}>
        
        <h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#1d4ed8', marginBottom: '1.5rem' }}>Upload MRI Scan</h2>

        <form onSubmit={handleProcessScan}>
          <div className="form-group" style={{ textAlign: 'left', marginBottom: '1.5rem' }}>
            <label className="form-label" htmlFor="patientId">Patient ID</label>
            <input 
              type="text" 
              id="patientId" 
              className="form-input" 
              placeholder="e.g. A6740A34" 
              value={patientId}
              onChange={(e) => setPatientId(e.target.value.toUpperCase())}
              required 
            />
          </div>

          <div style={{ 
            border: '2px dashed #93c5fd', 
            borderRadius: '8px', 
            padding: '2rem 1rem', 
            backgroundColor: '#eff6ff',
            marginBottom: '1.5rem',
            cursor: 'pointer',
            position: 'relative'
          }}>
            <input 
              type="file" 
              multiple 
              onChange={handleFileChange}
              style={{ position: 'absolute', inset: 0, opacity: 0, cursor: 'pointer' }}
              accept=".dcm,.png,.jpg,.jpeg"
            />
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', color: '#2563eb' }}>
              <UploadCloud size={32} style={{ marginBottom: '0.5rem' }} />
              <p style={{ fontWeight: '500' }}>Drag & Drop DICOM or Image Files</p>
              <p style={{ fontSize: '0.875rem', color: '#60a5fa', marginTop: '0.25rem' }}>or click to browse</p>
            </div>
          </div>

          {files.length > 0 && (
            <div style={{ textAlign: 'left', marginBottom: '1.5rem', maxHeight: '100px', overflowY: 'auto' }}>
              {files.map((file, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.5rem', backgroundColor: '#f8fafc', borderRadius: '4px', marginBottom: '0.25rem', fontSize: '0.875rem', color: '#475569' }}>
                  <File size={16} color="#94a3b8" />
                  <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{file.name}</span>
                  <CheckCircle size={14} color="#22c55e" style={{ marginLeft: 'auto' }} />
                </div>
              ))}
            </div>
          )}

          <button 
            type="submit" 
            className="btn-primary" 
            style={{ width: '100%', padding: '0.75rem', fontSize: '1rem' }}
            disabled={isUploading}
          >
            {isUploading ? 'Processing...' : 'Process Scan'}
          </button>
          
          <p style={{ textAlign: 'center', fontSize: '0.75rem', color: '#94a3b8', marginTop: '1rem' }}>
            Your session is fully encrypted and HIPAA-compliant.
          </p>
        </form>
      </div>
    </div>
  );
}
