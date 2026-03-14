import { useNavigate } from 'react-router-dom';
import { Brain } from 'lucide-react';
import { dataStore } from '../lib/dummyData';

export default function Signup({ onSignup }) {
  const navigate = useNavigate();

  const handleSubmit = (e) => {
    e.preventDefault();
    
    // In a real app we'd capture data and send to API
    const formData = new FormData(e.target);
    dataStore.updateDoctor({
        name: formData.get('name'),
        email: formData.get('email'),
        hospital: formData.get('hospital')
    });
    
    onSignup();
    navigate('/dashboard');
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)', padding: '2rem 1rem' }}>
      <div style={{ background: 'white', padding: '3rem', borderRadius: '16px', width: '100%', maxWidth: '450px', boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)' }}>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginBottom: '2rem' }}>
          <div style={{ backgroundColor: '#eff6ff', padding: '1rem', borderRadius: '50%', marginBottom: '1rem' }}>
            <Brain size={48} color="#2563eb" />
          </div>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#1e293b' }}>Doctor Sign Up</h2>
          <p style={{ color: '#64748b', marginTop: '0.5rem', textAlign: 'center' }}>Register for NeuroVision AI clinical access</p>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label className="form-label" htmlFor="name">Doctor Name</label>
            <input type="text" id="name" name="name" className="form-input" placeholder="e.g. Dr. John Smith" required />
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="email">Email</label>
            <input type="email" id="email" name="email" className="form-input" placeholder="doctor@hospital.org" required />
          </div>
          
          <div className="form-group">
            <label className="form-label" htmlFor="password">Password</label>
            <input type="password" id="password" name="password" className="form-input" placeholder="Create a strong password" required />
          </div>

          <div className="form-group" style={{ marginBottom: '2rem' }}>
            <label className="form-label" htmlFor="hospital">Hospital Name</label>
            <input type="text" id="hospital" name="hospital" className="form-input" placeholder="e.g. General Hospital" required />
          </div>

          <button type="submit" className="btn-primary" style={{ width: '100%', padding: '0.75rem', fontSize: '1rem' }}>
            Create Account
          </button>
        </form>
        
        <div style={{ marginTop: '1.5rem', textAlign: 'center' }}>
          <p style={{ color: '#64748b', fontSize: '0.875rem' }}>
            Already have an account? <a href="/login" style={{ color: '#2563eb', fontWeight: '500' }}>Login</a>
          </p>
        </div>
      </div>
    </div>
  );
}
