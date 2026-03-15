import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, Copy, Check } from 'lucide-react'; // Added Copy and Check

export default function Dashboard() {
  const navigate = useNavigate();
  const [searchTerm, setSearchTerm] = useState('');
  const [patients, setPatients] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [copiedId, setCopiedId] = useState(null); // Track which ID was copied

  const doctorName = localStorage.getItem('neurovision_doctor_name') || 'Doctor';

  useEffect(() => {
    const fetchPatients = async () => {
      const currentDoctorId = localStorage.getItem('neurovision_doctor_id');
      if (!currentDoctorId) {
        setIsLoading(false);
        return;
      }
      try {
        const response = await fetch(`http://127.0.0.1:8000/patients?doctor_id=${currentDoctorId}`);
        if (!response.ok) throw new Error('Failed to fetch patients');
        const data = await response.json();
        setPatients(data);
      } catch (error) {
        console.error("Error fetching patients:", error);
      } finally {
        setIsLoading(false);
      }
    };
    fetchPatients();
  }, []);

  const handleCopy = (id) => {
    navigator.clipboard.writeText(id);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const filteredPatients = patients.filter(p =>
    (p.name && p.name.toLowerCase().includes(searchTerm.toLowerCase())) ||
    (p.id && String(p.id).toLowerCase().includes(searchTerm.toLowerCase()))
  );

  return (
    <div>
      <div style={{ marginBottom: '2rem' }}>
        <h1 style={{ fontSize: '2rem', fontWeight: 'bold', color: '#0f172a' }}>Welcome back, {doctorName}</h1>
        <p style={{ color: '#64748b' }}>Overview: Your Clinical activity</p>
      </div>

      <div style={{ position: 'relative', marginBottom: '2rem', maxWidth: '600px' }}>
        <div style={{ position: 'absolute', left: '1rem', top: '50%', transform: 'translateY(-50%)', color: '#94a3b8' }}>
          <Search size={20} />
        </div>
        <input
          type="text"
          placeholder="Search patients by name or ID..."
          className="form-input"
          style={{ paddingLeft: '3rem', height: '3rem', borderRadius: '8px' }}
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
      </div>

      <div className="card">
        <h2 style={{ fontSize: '1.25rem', fontWeight: 'bold', marginBottom: '1.5rem', color: '#1e293b' }}>Recent Activity</h2>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
            <thead>
              <tr style={{ backgroundColor: '#f8fafc', borderBottom: '1px solid #e2e8f0' }}>
                <th style={{ padding: '1rem', color: '#475569', fontWeight: '600' }}>Patient Name</th>
                <th style={{ padding: '1rem', color: '#475569', fontWeight: '600' }}>Patient ID</th>
                <th style={{ padding: '1rem', color: '#475569', fontWeight: '600' }}>Gender</th>
                <th style={{ padding: '1rem', color: '#475569', fontWeight: '600' }}>Age</th>
                <th style={{ padding: '1rem', color: '#475569', fontWeight: '600' }}>Height</th>
                <th style={{ padding: '1rem', color: '#475569', fontWeight: '600' }}>Weight</th>
                <th style={{ padding: '1rem', color: '#475569', fontWeight: '600' }}>Action</th>
              </tr>
            </thead>
            <tbody>
              {isLoading ? (
                <tr><td colSpan="7" style={{ padding: '2rem', textAlign: 'center' }}>Loading...</td></tr>
              ) : filteredPatients.map((patient) => (
                <tr key={patient.id} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '1rem', fontWeight: '500' }}>{patient.name}</td>
                  <td style={{ padding: '1rem', color: '#64748b', fontFamily: 'monospace', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span>{String(patient.id).substring(0, 8)}</span>
                    <button
                      onClick={() => handleCopy(patient.id)}
                      style={{ background: 'none', border: 'none', cursor: 'pointer', color: copiedId === patient.id ? '#22c55e' : '#94a3b8' }}
                      title="Copy Full UUID"
                    >
                      {copiedId === patient.id ? <Check size={14} /> : <Copy size={14} />}
                    </button>
                  </td>
                  <td style={{ padding: '1rem' }}>{patient.gender}</td>
                  <td style={{ padding: '1rem' }}>{patient.age}y</td>
                  <td style={{ padding: '1rem' }}>{patient.height} cm</td>
                  <td style={{ padding: '1rem' }}>{patient.weight} kg</td>
                  <td style={{ padding: '1rem' }}>
                    <button className="btn-primary" onClick={() => navigate(`/dashboard/patient/${patient.id}`)}>View History</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}