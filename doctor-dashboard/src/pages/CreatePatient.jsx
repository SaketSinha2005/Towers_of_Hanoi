import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { User, Calendar, Phone, Scale, Ruler } from 'lucide-react';

export default function CreatePatient() {
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const [formData, setFormData] = useState({
    name: '',
    dateOfBirth: '',
    gender: 'Male',
    countryCode: '+1',
    phone: '',
    height: '',
    weight: ''
  });

  const calculateAge = (dob) => {
    if(!dob) return '';
    const diff_ms = Date.now() - new Date(dob).getTime();
    const age_dt = new Date(diff_ms);
    return Math.abs(age_dt.getUTCFullYear() - 1970);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    const age = calculateAge(formData.dateOfBirth) || 30;
    const fullPhone = `${formData.countryCode} ${formData.phone}`;
    const currentDoctorId = localStorage.getItem('neurovision_doctor_id');
    // In a full app, you would get this from your logged-in user context/session
    if (!currentDoctorId) {
      setError("Doctor session not found. Please log out and log back in.");
      setIsLoading(false);
      return;
    }

    try {
      // Build the URL with query parameters to match FastAPI's expectation
      const url = new URL('http://127.0.0.1:8000/patients');
      url.searchParams.append('name', formData.name);
      url.searchParams.append('age', age);
      url.searchParams.append('gender', formData.gender);
      url.searchParams.append('phone', fullPhone);
      url.searchParams.append('height', formData.height);
      url.searchParams.append('weight', formData.weight);
      url.searchParams.append('doctor_id', currentDoctorId);

      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Accept': 'application/json' }
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || 'Failed to create patient');
      }

      console.log("Patient created with custom ID:", data.patient[0].id);

      // Redirect to dashboard on success
      navigate('/dashboard');

    } catch (err) {
      console.error("Error creating patient:", err);
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '600px', margin: '0 auto', display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 'calc(100vh - 4rem)' }}>
      <div className="card" style={{ width: '100%', padding: '2.5rem', border: '1px solid #e2e8f0', boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)' }}>
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#1e293b', marginBottom: '0.5rem' }}>Access Your NeuroVision Patient Hub</h2>
          <p style={{ color: '#64748b', fontSize: '0.875rem' }}>Please populate all fields with accurate clinical data.</p>
        </div>

        {error && (
          <div style={{ backgroundColor: '#fee2e2', color: '#b91c1c', padding: '0.75rem', borderRadius: '8px', marginBottom: '1.5rem', fontSize: '0.875rem', textAlign: 'center' }}>
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <User size={16} /> Patient Full Name
            </label>
            <input
              type="text"
              className="form-input"
              placeholder="e.g. Johnathan Doe"
              value={formData.name}
              onChange={(e) => setFormData({...formData, name: e.target.value})}
              required
            />
          </div>

          <div style={{ display: 'flex', gap: '1rem' }}>
            <div className="form-group" style={{ flex: 1 }}>
              <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Calendar size={16} /> Date of Birth
              </label>
              <input
                type="date"
                className="form-input"
                value={formData.dateOfBirth}
                onChange={(e) => setFormData({...formData, dateOfBirth: e.target.value})}
                required
              />
            </div>
            <div style={{ display: 'flex', alignItems: 'flex-end', paddingBottom: '1rem', width: '100px' }}>
              <div style={{ padding: '0.6rem', backgroundColor: '#dcfce7', color: '#166534', borderRadius: '6px', width: '100%', textAlign: 'center', fontSize: '0.875rem', fontWeight: '500' }}>
                Age: {calculateAge(formData.dateOfBirth) || '-'}
              </div>
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">Gender</label>
            <div style={{ display: 'flex', gap: '1rem' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.5rem 1rem', border: `1px solid ${formData.gender === 'Male' ? '#2563eb' : '#e2e8f0'}`, borderRadius: '6px', backgroundColor: formData.gender === 'Male' ? '#eff6ff' : 'white', cursor: 'pointer', flex: 1 }}>
                <input type="radio" name="gender" value="Male" checked={formData.gender === 'Male'} onChange={(e) => setFormData({...formData, gender: e.target.value})} style={{ display: 'none' }} />
                Male
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.5rem 1rem', border: `1px solid ${formData.gender === 'Female' ? '#2563eb' : '#e2e8f0'}`, borderRadius: '6px', backgroundColor: formData.gender === 'Female' ? '#eff6ff' : 'white', cursor: 'pointer', flex: 1 }}>
                <input type="radio" name="gender" value="Female" checked={formData.gender === 'Female'} onChange={(e) => setFormData({...formData, gender: e.target.value})} style={{ display: 'none' }} />
                Female
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.5rem 1rem', border: `1px solid ${formData.gender === 'Other' ? '#2563eb' : '#e2e8f0'}`, borderRadius: '6px', backgroundColor: formData.gender === 'Other' ? '#eff6ff' : 'white', cursor: 'pointer', flex: 1 }}>
                <input type="radio" name="gender" value="Other" checked={formData.gender === 'Other'} onChange={(e) => setFormData({...formData, gender: e.target.value})} style={{ display: 'none' }} />
                Other
              </label>
            </div>
          </div>

          <div className="form-group">
            <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Phone size={16} /> Phone Number
            </label>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <select
                className="form-input"
                style={{ width: '130px', padding: '0.5rem' }}
                value={formData.countryCode}
                onChange={(e) => setFormData({...formData, countryCode: e.target.value})}
              >
                <option value="+1">+1 (US/CA)</option>
                <option value="+44">+44 (UK)</option>
                <option value="+91">+91 (IN)</option>
                <option value="+61">+61 (AU)</option>
                <option value="+81">+81 (JP)</option>
              </select>
              <input
                type="tel"
                className="form-input"
                placeholder="(555) 012-3456"
                value={formData.phone}
                onChange={(e) => setFormData({...formData, phone: e.target.value})}
                required
                style={{ flex: 1 }}
              />
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '2rem' }}>
            <div className="form-group">
              <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Ruler size={16} /> Height (cm)
              </label>
              <input
                type="text"
                className="form-input"
                placeholder="175"
                value={formData.height}
                onChange={(e) => setFormData({...formData, height: e.target.value})}
                required
              />
            </div>

            <div className="form-group">
              <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Scale size={16} /> Weight (kg)
              </label>
              <input
                type="text"
                className="form-input"
                placeholder="70"
                value={formData.weight}
                onChange={(e) => setFormData({...formData, weight: e.target.value})}
                required
              />
            </div>
          </div>

          <div style={{ display: 'flex', gap: '1rem', justifyContent: 'flex-end', borderTop: '1px solid #e2e8f0', paddingTop: '1.5rem', marginTop: '1rem' }}>
            <button type="button" className="btn-outline" onClick={() => navigate('/dashboard')}>
              Cancel
            </button>
            <button type="submit" className="btn-primary" disabled={isLoading}>
              {isLoading ? 'Processing...' : 'Create Profile'}
            </button>
          </div>
          <p style={{ textAlign: 'center', fontSize: '0.75rem', color: '#94a3b8', marginTop: '1rem' }}>
            Your session is fully encrypted and HIPAA-compliant.
          </p>
        </form>
      </div>
    </div>
  );
}