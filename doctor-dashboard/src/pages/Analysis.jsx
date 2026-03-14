import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Send, BrainCircuit, Activity, ChevronRight } from 'lucide-react';
import { dataStore } from '../lib/dummyData';

export default function Analysis() {
  const { patientId } = useParams();
  const navigate = useNavigate();
  const patient = dataStore.getPatient(patientId);
  
  if (!patient) {
    return <div style={{ padding: '2rem' }}>Patient not found.</div>;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <button 
            onClick={() => navigate('/dashboard')} 
            style={{ padding: '0.5rem', borderRadius: '50%', backgroundColor: '#e2e8f0', color: '#475569', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
          >
            <ArrowLeft size={20} />
          </button>
          <div>
            <h1 style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#0f172a' }}>AI Diagnostics Report</h1>
            <p style={{ color: '#64748b', fontSize: '0.875rem' }}>Patient: {patient.name} ({patient.id})</p>
          </div>
        </div>
        
        <button className="btn-primary" onClick={() => navigate('/dashboard')}>
          Exit Analysis
        </button>
      </div>

      {/* Main Content Area */}
      <div style={{ display: 'flex', gap: '1.5rem', flex: 1, minHeight: 0 }}>
        
        {/* Left Side (70%) - Images & Report */}
        <div className="card" style={{ flex: '7', display: 'flex', flexDirection: 'column', overflowY: 'auto' }}>
          <h2 style={{ fontSize: '1.25rem', fontWeight: 'bold', marginBottom: '1rem', color: '#1e293b', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Activity size={20} color="#2563eb" /> Scan Results
          </h2>
          
          {/* Mock Images Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '1rem', marginBottom: '2rem' }}>
            {[1, 2, 3, 4, 5].map((num) => (
              <div key={num} style={{ aspectRatio: '1', backgroundColor: '#e2e8f0', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative', overflow: 'hidden' }}>
                <div style={{ position: 'absolute', inset: 0, backgroundColor: '#0f172a', opacity: 0.8 }}></div>
                <BrainCircuit size={48} color="#94a3b8" style={{ zIndex: 1 }} />
                <span style={{ position: 'absolute', bottom: '0.5rem', left: '0.5rem', color: 'white', fontSize: '0.75rem', zIndex: 1, fontWeight: '500' }}>MRI_Slice_0{num}</span>
              </div>
            ))}
          </div>

          {/* AI Text Analysis */}
          <div style={{ backgroundColor: '#f8fafc', padding: '1.5rem', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 'bold', color: '#0f172a', marginBottom: '1rem' }}>AI Generated Findings</h3>
            <div style={{ color: '#334155', lineHeight: '1.6' }}>
              <p style={{ marginBottom: '1rem' }}>
                <strong style={{ color: '#1d4ed8' }}>Automated Assessment:</strong> No acute intracranial abnormalities detected. The ventricular system and basal cisterns are unremarkable. Neural tissue appears within normal limits for the patient's age demographic.
              </p>
              <ul style={{ paddingLeft: '1.5rem', marginBottom: '1rem', listStyleType: 'disc' }}>
                <li>No evidence of ischemia, hemorrhage, or mass lesion.</li>
                <li>Midline shift: None (0.0mm).</li>
                <li>White matter appears confluent with no hyperintensities.</li>
              </ul>
              <div style={{ display: 'inline-block', backgroundColor: '#dcfce7', color: '#166534', padding: '0.5rem 1rem', borderRadius: '4px', fontSize: '0.875rem', fontWeight: '500' }}>
                Confidence Score: 98.4%
              </div>
            </div>
          </div>
        </div>

        {/* Right Side (30%) - Interactive Chatbox */}
        <div className="card" style={{ flex: '3', display: 'flex', flexDirection: 'column' }}>
          <h2 style={{ fontSize: '1.25rem', fontWeight: 'bold', marginBottom: '1rem', color: '#1e293b', display: 'flex', alignItems: 'center', gap: '0.5rem', borderBottom: '1px solid #e2e8f0', paddingBottom: '1rem' }}>
            <BrainCircuit size={20} color="#8b5cf6" /> NeuroVision Copilot
          </h2>
          
          <div style={{ flex: 1, overflowY: 'auto', paddingRight: '0.5rem', display: 'flex', flexDirection: 'column', gap: '1rem', marginBottom: '1rem' }}>
            <div style={{ alignSelf: 'flex-start', backgroundColor: '#f1f5f9', color: '#334155', padding: '0.75rem 1rem', borderRadius: '0.5rem 0.5rem 0.5rem 0', maxWidth: '85%' }}>
              <p style={{ fontSize: '0.9rem' }}>Hello Dr. I have completed the preliminary analysis of {patient.name}'s recent MRI. How can I assist you with these findings?</p>
            </div>
            
            <div style={{ alignSelf: 'flex-end', backgroundColor: '#2563eb', color: 'white', padding: '0.75rem 1rem', borderRadius: '0.5rem 0.5rem 0 0.5rem', maxWidth: '85%' }}>
              <p style={{ fontSize: '0.9rem' }}>Can you compare the ventricular volume to the previous baseline scan from January?</p>
            </div>
            
            <div style={{ alignSelf: 'flex-start', backgroundColor: '#f1f5f9', color: '#334155', padding: '0.75rem 1rem', borderRadius: '0.5rem 0.5rem 0.5rem 0', maxWidth: '85%' }}>
              <p style={{ fontSize: '0.9rem' }}>Calculating volumetric differences... The current lateral ventricular volume is 24.1 cm³. This represents a 0.5% variance from the January baseline of 24.0 cm³, which is statistically insignificant and within the margin of error.</p>
            </div>
          </div>

          <div style={{ position: 'relative', marginTop: 'auto' }}>
            <input 
              type="text" 
              className="form-input" 
              placeholder="Ask Copilot about the scan..." 
              style={{ paddingRight: '3rem', borderRadius: '20px' }}
            />
            <button style={{ position: 'absolute', right: '0.5rem', top: '50%', transform: 'translateY(-50%)', backgroundColor: '#2563eb', color: 'white', padding: '0.4rem', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Send size={16} />
            </button>
          </div>
        </div>

      </div>
    </div>
  );
}
