import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search } from 'lucide-react';
import { dataStore } from '../lib/dummyData';

export default function Dashboard() {
  const navigate = useNavigate();
  const [searchTerm, setSearchTerm] = useState('');
  const doctor = dataStore.getDoctor();
  const patients = dataStore.getPatients();

  const filteredPatients = patients.filter(p => 
    p.name.toLowerCase().includes(searchTerm.toLowerCase()) || 
    p.id.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const getStatusBadge = (status) => {
    switch(status) {
      case 'Analysis Complete':
        return <span className="badge badge-success">{status}</span>;
      case 'Processing':
        return <span className="badge badge-warning">{status}</span>;
      case 'Requires Review':
        return <span className="badge badge-danger">{status}</span>;
      default:
        return <span className="badge" style={{ backgroundColor: '#e2e8f0', color: '#475569' }}>{status}</span>;
    }
  };

  return (
    <div>
      <div style={{ marginBottom: '2rem' }}>
        <h1 style={{ fontSize: '2rem', fontWeight: 'bold', color: '#0f172a' }}>Welcome back, {doctor.name}</h1>
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
                <th style={{ padding: '1rem', color: '#475569', fontWeight: '600', fontSize: '0.875rem' }}>Patient Name</th>
                <th style={{ padding: '1rem', color: '#475569', fontWeight: '600', fontSize: '0.875rem' }}>Medical ID</th>
                <th style={{ padding: '1rem', color: '#475569', fontWeight: '600', fontSize: '0.875rem' }}>Gender / Age</th>
                <th style={{ padding: '1rem', color: '#475569', fontWeight: '600', fontSize: '0.875rem' }}>Last MRI Date</th>
                <th style={{ padding: '1rem', color: '#475569', fontWeight: '600', fontSize: '0.875rem' }}>Current Status</th>
                <th style={{ padding: '1rem', color: '#475569', fontWeight: '600', fontSize: '0.875rem' }}>Action</th>
              </tr>
            </thead>
            <tbody>
              {filteredPatients.map((patient) => (
                <tr key={patient.id} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '1rem', fontWeight: '500', color: '#0f172a' }}>{patient.name}</td>
                  <td style={{ padding: '1rem', color: '#64748b', fontFamily: 'monospace' }}>{patient.id}</td>
                  <td style={{ padding: '1rem', color: '#64748b' }}>{patient.gender}, {patient.age}</td>
                  <td style={{ padding: '1rem', color: '#64748b' }}>{patient.lastMriDate || '-'}</td>
                  <td style={{ padding: '1rem' }}>{getStatusBadge(patient.status)}</td>
                  <td style={{ padding: '1rem' }}>
                    <button 
                      className="btn-primary" 
                      style={{ padding: '0.4rem 0.75rem', fontSize: '0.875rem' }}
                      onClick={() => navigate(`/dashboard/patient/${patient.id}`)}
                    >
                      View History
                    </button>
                  </td>
                </tr>
              ))}
              {filteredPatients.length === 0 && (
                <tr>
                  <td colSpan="6" style={{ padding: '2rem', textAlign: 'center', color: '#64748b' }}>
                    No patients found matching your search.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
