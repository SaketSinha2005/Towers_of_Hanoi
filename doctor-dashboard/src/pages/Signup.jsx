import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Brain } from 'lucide-react';

export default function Signup({ onSignup }) {
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    const formData = new FormData(e.target);
    const name = formData.get('name');
    const email = formData.get('email');
    const password = formData.get('password');
    const hospital = formData.get('hospital');

    try {
      const response = await fetch('http://127.0.0.1:8000/doctor/signup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify({ name, email, password, hospital })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || data.message || 'Registration failed');
      }

      console.log("Signup successful:", data);

      // Save to localStorage so they are immediately recognized by the Dashboard
      localStorage.setItem('neurovision_doctor_id', data.doctor_id);
      localStorage.setItem('neurovision_doctor_name', name.startsWith('Dr.') ? name : `Dr. ${name}`);

      onSignup();
      navigate('/dashboard');

    } catch (err) {
      console.error("Signup Error:", err);
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
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

        {error && (
          <div style={{ backgroundColor: '#fee2e2', color: '#b91c1c', padding: '0.75rem', borderRadius: '8px', marginBottom: '1.5rem', fontSize: '0.875rem', textAlign: 'center' }}>
            {error}
          </div>
        )}

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

          <button type="submit" className="btn-primary" style={{ width: '100%', padding: '0.75rem', fontSize: '1rem' }} disabled={isLoading}>
            {isLoading ? 'Creating Account...' : 'Create Account'}
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