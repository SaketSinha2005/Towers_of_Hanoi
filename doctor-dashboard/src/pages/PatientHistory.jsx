import { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Calendar, FileText, Download, ChevronDown, ChevronUp } from 'lucide-react';
import { dataStore } from '../lib/dummyData';

export default function PatientHistory() {
  const { patientId } = useParams();
  const navigate = useNavigate();
  const patient = dataStore.getPatient(patientId);
  const checkupsList = dataStore.getCheckups(patientId);
  
  const [expandedCheckups, setExpandedCheckups] = useState(
    checkupsList.reduce((acc, curr, idx) => ({ ...acc, [curr.id]: idx === 0 }), {}) // Open first by default
  );

  const toggleCheckup = (id) => {
    setExpandedCheckups(prev => ({ ...prev, [id]: !prev[id] }));
  };

  if (!patient) {
    return <div style={{ padding: '2rem' }}>Patient not found.</div>;
  }

  return (
    <div style={{ maxWidth: '900px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      
      {/* Header Controls */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        <button 
          onClick={() => navigate('/dashboard')} 
          style={{ padding: '0.5rem', borderRadius: '50%', backgroundColor: '#e2e8f0', color: '#475569', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
        >
          <ArrowLeft size={20} />
        </button>
        <h1 style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#0f172a' }}>Patient Details & History</h1>
      </div>

      {/* Patient Profile Card */}
      <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem', borderLeft: '4px solid #3b82f6' }}>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#0f172a' }}>{patient.name}</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '1.5rem', marginTop: '0.5rem' }}>
          <div>
            <p style={{ color: '#64748b', fontSize: '0.875rem', marginBottom: '0.25rem' }}>Medical ID</p>
            <p style={{ fontWeight: '600', color: '#1e293b', fontFamily: 'monospace' }}>{patient.id}</p>
          </div>
          <div>
            <p style={{ color: '#64748b', fontSize: '0.875rem', marginBottom: '0.25rem' }}>Age / Gender</p>
            <p style={{ fontWeight: '500', color: '#1e293b' }}>{patient.age} years / {patient.gender}</p>
          </div>
          <div>
            <p style={{ color: '#64748b', fontSize: '0.875rem', marginBottom: '0.25rem' }}>Vitals (H / W)</p>
            <p style={{ fontWeight: '500', color: '#1e293b' }}>{patient.height} / {patient.weight}</p>
          </div>
          <div>
            <p style={{ color: '#64748b', fontSize: '0.875rem', marginBottom: '0.25rem' }}>Contact Info</p>
            <p style={{ fontWeight: '500', color: '#1e293b' }}>{patient.phone}</p>
          </div>
        </div>
      </div>

      {/* Checkups History */}
      <div>
        <h2 style={{ fontSize: '1.25rem', fontWeight: 'bold', color: '#0f172a', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Calendar size={20} color="#64748b" /> Clinical History
        </h2>

        {checkupsList.length === 0 ? (
          <div className="card" style={{ textAlign: 'center', padding: '3rem', color: '#64748b' }}>
            No clinical records or MRI scans found for this patient.
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {checkupsList.map((checkup) => (
              <div key={checkup.id} className="card" style={{ padding: 0, overflow: 'hidden' }}>
                {/* Accordion Header */}
                <div 
                  style={{ padding: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer', backgroundColor: expandedCheckups[checkup.id] ? '#f8fafc' : 'white' }}
                  onClick={() => toggleCheckup(checkup.id)}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <div style={{ backgroundColor: '#e0e7ff', padding: '0.75rem', borderRadius: '8px', color: '#4f46e5' }}>
                      <FileText size={24} />
                    </div>
                    <div>
                      <h3 style={{ fontSize: '1.1rem', fontWeight: '600', color: '#1e293b' }}>MRI Assessment</h3>
                      <p style={{ color: '#64748b', fontSize: '0.875rem', marginTop: '0.25rem' }}>Date: {checkup.date}</p>
                    </div>
                  </div>
                  <div>
                    {expandedCheckups[checkup.id] ? <ChevronUp size={24} color="#64748b" /> : <ChevronDown size={24} color="#64748b" />}
                  </div>
                </div>

                {/* Accordion Body */}
                {expandedCheckups[checkup.id] && (
                  <div style={{ padding: '1.5rem', borderTop: '1px solid #e2e8f0', backgroundColor: 'white' }}>
                    <div style={{ marginBottom: '1.5rem' }}>
                      <h4 style={{ fontSize: '0.875rem', fontWeight: '600', color: '#475569', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.75rem' }}>Scans Uploaded</h4>
                      <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                        {checkup.scans.map((scan, i) => (
                          <div key={i} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem', backgroundColor: '#f1f5f9', padding: '0.5rem 1rem', borderRadius: '6px', fontSize: '0.875rem', color: '#334155' }}>
                            <FileText size={16} color="#64748b" />
                            {scan}
                          </div>
                        ))}
                      </div>
                    </div>

                    <div style={{ marginBottom: '1.5rem' }}>
                      <h4 style={{ fontSize: '0.875rem', fontWeight: '600', color: '#475569', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.75rem' }}>AI Text Analysis</h4>
                      <p style={{ color: '#334155', lineHeight: '1.6', backgroundColor: '#f8fafc', padding: '1rem', borderRadius: '8px', borderLeft: '3px solid #cbd5e1' }}>
                        {checkup.analysis}
                      </p>
                    </div>

                    <div style={{ display: 'flex', gap: '1rem' }}>
                      <button className="btn-outline" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }} onClick={(e) => { e.stopPropagation(); navigate(`/dashboard/analysis/${patient.id}`); }}>
                        Open in Copilot <ArrowLeft size={16} style={{ rotate: '180deg' }}/>
                      </button>
                      <button className="btn-primary" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', backgroundColor: '#f1f5f9', color: '#0f172a', border: '1px solid #cbd5e1' }}>
                        <Download size={16} /> Download Full Report
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
