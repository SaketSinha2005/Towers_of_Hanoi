import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Calendar, FileText, Download, ChevronDown, ChevronUp, Copy, Check } from 'lucide-react';

export default function PatientHistory() {
  const { patientId } = useParams();
  const navigate = useNavigate();

  const [patient, setPatient] = useState(null);
  const [allScans, setAllScans] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchScanId, setSearchScanId] = useState('');
  const [expandedScans, setExpandedScans] = useState({});
  const [copiedId, setCopiedId] = useState(null);

// PatientHistory.jsx
// Inside PatientHistory.jsx - Corrected useEffect logic

useEffect(() => {
  const fetchPatientData = async () => {
    try {
      const patientRes = await fetch(`http://127.0.0.1:8000/patient/${patientId}`);
      if (!patientRes.ok) throw new Error("Patient not found");
      const data = await patientRes.json();
      setPatient(data);

      const scansRes = await fetch(`http://127.0.0.1:8000/patient/${patientId}/scans`);
      let scansData = scansRes.ok ? await scansRes.json() : [];

      // Fetch analyses for each scan
      const scansWithAnalysis = await Promise.all(scansData.map(async (scan) => {
        try {
          // FIX: Perform the actual fetch call
          const analysisRes = await fetch(`http://127.0.0.1:8000/scan/${scan.id}/analysis`);
          const analysisData = await analysisRes.json();

          // FIX: Check for the 'explanation' field directly (not .length)
          const reportText = (analysisData && analysisData.explanation)
            ? analysisData.explanation
            : "Analysis results are still being processed.";

          return { ...scan, report: reportText };
        } catch (e) {
          console.error("Analysis fetch error for scan:", scan.id, e);
          return { ...scan, report: "Error loading analysis data." };
        }
      }));

      setAllScans(scansWithAnalysis);
    } catch (error) {
      console.error("Clinical data fetch error:", error);
    } finally {
      setIsLoading(false);
    }
  };
  fetchPatientData();
}, [patientId]);

  const handleCopy = (id) => {
    navigator.clipboard.writeText(id);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const toggleScan = (id) => setExpandedScans(prev => ({ ...prev, [id]: !prev[id] }));

  const scansList = searchScanId.trim() ? allScans.filter(s => String(s.id).toLowerCase().includes(searchScanId.toLowerCase())) : allScans;

  if (isLoading) return <div style={{ padding: '2rem', textAlign: 'center' }}>Loading...</div>;
  if (!patient || patient.error) return <div style={{ padding: '2rem', textAlign: 'center', color: '#b91c1c' }}>Patient not found.</div>;

  return (
    <div style={{ maxWidth: '900px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        <button onClick={() => navigate('/dashboard')} style={{ padding: '0.5rem', borderRadius: '50%', backgroundColor: '#e2e8f0', display: 'flex' }}><ArrowLeft size={20} /></button>
        <h1 style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>Patient Details & History</h1>
      </div>

      <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem', borderLeft: '4px solid #3b82f6' }}>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{patient.name}</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '1.5rem' }}>
          <div>
            <p style={{ color: '#64748b', fontSize: '0.875rem' }}>Medical ID</p>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <p style={{ fontWeight: '600', fontFamily: 'monospace' }}>{patient.id}</p>
              <button onClick={() => handleCopy(patient.id)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: copiedId === patient.id ? '#22c55e' : '#94a3b8' }}>
                {copiedId === patient.id ? <Check size={14} /> : <Copy size={14} />}
              </button>
            </div>
          </div>
          <div><p style={{ color: '#64748b', fontSize: '0.875rem' }}>Age / Gender</p><p>{patient.age}y / {patient.gender}</p></div>
          <div><p style={{ color: '#64748b', fontSize: '0.875rem' }}>Vitals</p><p>{patient.height}cm / {patient.weight}kg</p></div>
        </div>
      </div>

      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem' }}>
          <h2 style={{ fontSize: '1.25rem', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '0.5rem' }}><Calendar size={20} /> Scan History</h2>
          <input type="text" placeholder="Search by Scan ID..." className="form-input" style={{ width: '250px' }} value={searchScanId} onChange={(e) => setSearchScanId(e.target.value)} />
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {scansList.map((scan) => (
            <div key={scan.id} className="card" style={{ padding: 0, overflow: 'hidden' }}>
              <div style={{ padding: '1.5rem', display: 'flex', justifyContent: 'space-between', cursor: 'pointer' }} onClick={() => toggleScan(scan.id)}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <div style={{ backgroundColor: '#e0e7ff', padding: '0.75rem', borderRadius: '8px' }}><FileText size={24} /></div>
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <h3 style={{ fontWeight: '600' }}>Scan ID: {String(scan.id).substring(0, 8)}</h3>
                      <button
                        onClick={(e) => { e.stopPropagation(); handleCopy(scan.id); }}
                        style={{ background: 'none', border: 'none', cursor: 'pointer', color: copiedId === scan.id ? '#22c55e' : '#94a3b8' }}
                      >
                        {copiedId === scan.id ? <Check size={14} /> : <Copy size={14} />}
                      </button>
                    </div>
                    <p style={{ color: '#64748b', fontSize: '0.875rem' }}>{new Date(scan.uploaded_at).toLocaleDateString()}</p>
                  </div>
                </div>
                {expandedScans[scan.id] ? <ChevronUp size={24} /> : <ChevronDown size={24} />}
              </div>

              {expandedScans[scan.id] && (
                <div style={{ padding: '1.5rem', borderTop: '1px solid #e2e8f0' }}>
                  <div style={{ marginBottom: '1.5rem' }}>
                    <h4 style={{ fontSize: '0.875rem', fontWeight: '600', color: '#475569', marginBottom: '0.5rem' }}>AI ANALYSIS</h4>
                    <p style={{ backgroundColor: '#f8fafc', padding: '1rem', borderRadius: '8px' }}>{scan.report}</p>
                  </div>
                  <div style={{ display: 'flex', gap: '1rem' }}>
                    <button className="btn-outline" onClick={() => navigate(`/dashboard/analysis/${patient.id}`)}>Chat with AI</button>
                    <button className="btn-primary" onClick={() => window.location.href = `http://127.0.0.1:8000/scan/${scan.id}/report/download`}><Download size={16} /> Download PDF</button>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}