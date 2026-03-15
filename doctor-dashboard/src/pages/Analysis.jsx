import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Send, BrainCircuit, Activity } from 'lucide-react';

export default function Analysis() {
  const { patientId } = useParams();
  const navigate = useNavigate();

  const [patient, setPatient] = useState(null);
  const [analysis, setAnalysis] = useState(null); // Added state for analysis data
  const [scanId, setScanId] = useState(null);
  const [question, setQuestion] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isPageLoading, setIsPageLoading] = useState(true);
  const [messages, setMessages] = useState([]);

  useEffect(() => {
    const fetchFullAnalysis = async () => {
      try {
        // 1. Fetch Patient Info
        const pRes = await fetch(`http://127.0.0.1:8000/patient/${patientId}`);
        const pData = await pRes.json();
        setPatient(pData);

        // 2. Fetch Scans to get the latest ID
        const sRes = await fetch(`http://127.0.0.1:8000/patient/${patientId}/scans`);
        const sData = await sRes.json();

        if (sData && sData.length > 0) {
          // Store the scan id so we can show the associated image
          setScanId(sData[0].id);

          // 3. Fetch the specific analysis for the latest scan
          const aRes = await fetch(`http://127.0.0.1:8000/scan/${sData[0].id}/analysis`);
          const aData = await aRes.json();

          // Simplify: handle both single object or array responses
          const analysisObject = Array.isArray(aData) ? aData[0] : aData;

          if (analysisObject && analysisObject.explanation) {
            setAnalysis(analysisObject);
          }
        }

        setMessages([{
          role: 'ai',
          text: `MRI analysis for ${pData.name} is ready. I've detected ${sData.length} scans. How can I help?`
        }]);
      } catch (error) {
        console.error("Clinical fetch error:", error);
      } finally {
        setIsPageLoading(false);
      }
    };
    fetchFullAnalysis();
  }, [patientId]);

  const handleAskAI = async () => {
    if (!question.trim() || isLoading) return;
    const currentQ = question; setQuestion(''); setIsLoading(true);
    setMessages(prev => [...prev, { role: 'user', text: currentQ }]);

    try {
      const res = await fetch(`http://127.0.0.1:8000/ai/ask?question=${encodeURIComponent(currentQ)}`, { method: 'POST' });
      const data = await res.json();
      setMessages(prev => [...prev, { role: 'ai', text: data.answer }]);
    } catch (e) {
      setMessages(prev => [...prev, { role: 'ai', text: "Connection error with Copilot." }]);
    } finally {
      setIsLoading(false);
    }
  };

  if (isPageLoading) return <div style={{ padding: '2rem' }}>Loading Diagnostics...</div>;
  if (!patient) return <div style={{ padding: '2rem' }}>Patient record not found.</div>;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', padding: '1.5rem' }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <button onClick={() => navigate('/dashboard')} className="btn-outline" style={{ borderRadius: '50%', border: 'none', cursor: 'pointer', padding: '0.5rem' }}><ArrowLeft size={20} /></button>
          <h1 style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>Diagnostics Report: {patient.name}</h1>
        </div>
        <div style={{ display: 'flex', gap: '0.75rem' }}>
          {scanId && (
            <button className="btn-outline" onClick={() => window.open(`http://127.0.0.1:8000/scan/${scanId}/generate-image`, '_blank')}>
              View Result Image
            </button>
          )}
          <button className="btn-primary" onClick={() => navigate('/dashboard')}>Exit Analysis</button>
        </div>
      </header>

      <div style={{ display: 'flex', gap: '1.5rem', flex: 1, minHeight: 0 }}>
        {/* ENLARGED RESULTS AREA */}
        <div className="card" style={{ flex: '8', display: 'flex', flexDirection: 'column', overflowY: 'auto' }}>
          <h2 style={{ fontSize: '1.25rem', marginBottom: '1.5rem' }}><Activity size={20} color="#2563eb" style={{ marginRight: '8px' }} /> AI Analysis Findings</h2>

          {analysis ? (
            <div style={{ padding: '1rem' }}>
              <div style={{ backgroundColor: '#f8fafc', padding: '1.5rem', borderRadius: '12px', borderLeft: '4px solid #2563eb', marginBottom: '2rem' }}>
                <h3 style={{ fontWeight: 'bold', marginBottom: '1rem' }}>Clinical Interpretation</h3>
                <p style={{ lineHeight: '1.6', color: '#334155' }}>{analysis.explanation}</p>
              </div>

              {/* Volumetric Metrics Table */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem' }}>
                <div className="metric-box" style={{ padding: '1rem', background: '#eff6ff', borderRadius: '8px' }}>
                  <p style={{ fontSize: '0.8rem', color: '#1e40af' }}>Whole Tumor Volume</p>
                  <p style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>{analysis.tumor_volume}%</p>
                </div>
                <div className="metric-box" style={{ padding: '1rem', background: '#f0fdf4', borderRadius: '8px' }}>
                  <p style={{ fontSize: '0.8rem', color: '#166534' }}>Enhancing Ratio</p>
                  <p style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>{analysis.enhancing_ratio}%</p>
                </div>
                <div className="metric-box" style={{ padding: '1rem', background: '#fff7ed', borderRadius: '8px' }}>
                  <p style={{ fontSize: '0.8rem', color: '#9a3412' }}>Edema Ratio</p>
                  <p style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>{analysis.edema_ratio}%</p>
                </div>
              </div>
            </div>
          ) : (
            <div style={{ color: '#64748b', textAlign: 'center', padding: '6rem' }}>
              No active analysis found. Please upload an MRI scan to begin.
            </div>
          )}
        </div>

        {/* COPILOT SIDEBAR */}
        <div className="card" style={{ flex: '3', display: 'flex', flexDirection: 'column' }}>
          <h2 style={{ borderBottom: '1px solid #e2e8f0', paddingBottom: '1rem', marginBottom: '1rem' }}><BrainCircuit size={20} color="#8b5cf6" style={{ marginRight: '8px' }} /> NeuroVision Copilot</h2>
          <div style={{ flex: 1, overflowY: 'auto', marginBottom: '1rem' }}>
            {messages.map((m, i) => (
              <div key={i} style={{ alignSelf: m.role === 'ai' ? 'flex-start' : 'flex-end', backgroundColor: m.role === 'ai' ? '#f1f5f9' : '#2563eb', color: m.role === 'ai' ? 'black' : 'white', padding: '0.75rem', margin: '0.4rem', borderRadius: '8px', maxWidth: '90%' }}>
                <p style={{ fontSize: '0.9rem' }}>{m.text}</p>
              </div>
            ))}
            {isLoading && <p style={{ fontSize: '0.8rem', color: '#64748b', fontStyle: 'italic' }}>Thinking...</p>}
          </div>
          <div style={{ position: 'relative' }}>
            <input type="text" className="form-input" placeholder="Ask about findings..." value={question} onChange={(e) => setQuestion(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && handleAskAI()} />
            <button onClick={handleAskAI} style={{ position: 'absolute', right: '0.5rem', top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', cursor: 'pointer' }}><Send size={18} color="#2563eb" /></button>
          </div>
        </div>
      </div>
    </div>
  );
}