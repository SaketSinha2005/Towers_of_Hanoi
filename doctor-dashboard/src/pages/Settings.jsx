import { useState, useEffect } from 'react';
import { User, Lock, Mail, Building } from 'lucide-react';

export default function Settings() {
  const [doctor, setDoctor] = useState({ name: '', email: '', hospital: '' });
  const [passwordForm, setPasswordForm] = useState({ current: '', new: '', confirm: '' });
  const [saveMessage, setSaveMessage] = useState('');
  const [isLoading, setIsLoading] = useState(true);

  // Get the logged-in doctor's ID
  const doctorId = localStorage.getItem('neurovision_doctor_id');

  // Fetch the doctor's current details when the page loads
  useEffect(() => {
    const fetchDoctorDetails = async () => {
      if (!doctorId) {
        setIsLoading(false);
        return;
      }

      try {
        const response = await fetch(`http://127.0.0.1:8000/doctor/${doctorId}`);
        if (!response.ok) throw new Error('Failed to fetch profile');

        const data = await response.json();
        setDoctor({
          name: data.name || '',
          email: data.email || '',
          hospital: data.hospital || ''
        });
      } catch (error) {
        console.error("Error loading doctor profile:", error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchDoctorDetails();
  }, [doctorId]);

  const handleProfileSave = async (e) => {
    e.preventDefault();
    if (!doctorId) return;

    try {
      // Send the update to the backend using query params to match FastAPI
      const response = await fetch(`http://127.0.0.1:8000/doctor/${doctorId}?name=${encodeURIComponent(doctor.name)}&hospital=${encodeURIComponent(doctor.hospital)}`, {
        method: 'PUT',
        headers: { 'Accept': 'application/json' }
      });

      if (!response.ok) throw new Error('Failed to update profile');

      // Update local storage so the dashboard header changes immediately
      localStorage.setItem('neurovision_doctor_name', `Dr. ${doctor.name.split(' ')[0]}`);

      setSaveMessage('Profile saved successfully!');
      setTimeout(() => setSaveMessage(''), 3000);
    } catch (error) {
      alert("Error saving profile: " + error.message);
    }
  };

  const handlePasswordSave = async (e) => {
    e.preventDefault();

    if (passwordForm.new !== passwordForm.confirm) {
      alert("Passwords don't match!");
      return;
    }

    try {
      const response = await fetch(`http://127.0.0.1:8000/doctor/update-password?new_password=${encodeURIComponent(passwordForm.new)}`, {
        method: 'POST',
        headers: { 'Accept': 'application/json' }
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.message || 'Failed to update password');
      }

      setSaveMessage('Password updated successfully!');
      setPasswordForm({ current: '', new: '', confirm: '' });
      setTimeout(() => setSaveMessage(''), 3000);
    } catch (error) {
      alert("Error updating password: " + error.message);
    }
  };

  if (isLoading) {
    return <div style={{ padding: '2rem', textAlign: 'center', color: '#64748b' }}>Loading profile...</div>;
  }

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto' }}>
      <h1 style={{ fontSize: '2rem', fontWeight: 'bold', color: '#0f172a', marginBottom: '2rem' }}>Settings</h1>

      {saveMessage && (
        <div style={{ backgroundColor: '#dcfce7', color: '#166534', padding: '1rem', borderRadius: '8px', marginBottom: '2rem', fontWeight: '500' }}>
          {saveMessage}
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr)', gap: '2rem' }}>
        {/* Profile Settings */}
        <div className="card">
          <h2 style={{ fontSize: '1.25rem', fontWeight: 'bold', marginBottom: '1.5rem', color: '#1e293b', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <User size={20} /> Doctor Profile
          </h2>

          <form onSubmit={handleProfileSave}>
            <div className="form-group">
              <label className="form-label" htmlFor="name">Full Name</label>
              <div style={{ position: 'relative' }}>
                <span style={{ position: 'absolute', left: '0.75rem', top: '50%', transform: 'translateY(-50%)', color: '#94a3b8' }}><User size={18} /></span>
                <input
                  type="text"
                  id="name"
                  className="form-input"
                  style={{ paddingLeft: '2.5rem' }}
                  value={doctor.name}
                  onChange={(e) => setDoctor({...doctor, name: e.target.value})}
                  required
                />
              </div>
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="email">Email Address (Cannot be changed)</label>
              <div style={{ position: 'relative' }}>
                <span style={{ position: 'absolute', left: '0.75rem', top: '50%', transform: 'translateY(-50%)', color: '#94a3b8' }}><Mail size={18} /></span>
                <input
                  type="email"
                  id="email"
                  className="form-input"
                  style={{ paddingLeft: '2.5rem', backgroundColor: '#f1f5f9', color: '#64748b', cursor: 'not-allowed' }}
                  value={doctor.email}
                  readOnly
                  disabled
                />
              </div>
            </div>

            <div className="form-group" style={{ marginBottom: '2rem' }}>
              <label className="form-label" htmlFor="hospital">Hospital/Clinic Name</label>
              <div style={{ position: 'relative' }}>
                <span style={{ position: 'absolute', left: '0.75rem', top: '50%', transform: 'translateY(-50%)', color: '#94a3b8' }}><Building size={18} /></span>
                <input
                  type="text"
                  id="hospital"
                  className="form-input"
                  style={{ paddingLeft: '2.5rem' }}
                  value={doctor.hospital}
                  onChange={(e) => setDoctor({...doctor, hospital: e.target.value})}
                  required
                />
              </div>
            </div>

            <button type="submit" className="btn-primary">Save Profile</button>
          </form>
        </div>

        {/* Change Password */}
        <div className="card">
          <h2 style={{ fontSize: '1.25rem', fontWeight: 'bold', marginBottom: '1.5rem', color: '#1e293b', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Lock size={20} /> Change Password
          </h2>

          <form onSubmit={handlePasswordSave}>
            <div className="form-group">
              <label className="form-label" htmlFor="currentPassword">Current Password</label>
              <input
                type="password"
                id="currentPassword"
                className="form-input"
                value={passwordForm.current}
                onChange={(e) => setPasswordForm({...passwordForm, current: e.target.value})}
                required
              />
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '2rem' }}>
              <div className="form-group">
                <label className="form-label" htmlFor="newPassword">New Password</label>
                <input
                  type="password"
                  id="newPassword"
                  className="form-input"
                  value={passwordForm.new}
                  onChange={(e) => setPasswordForm({...passwordForm, new: e.target.value})}
                  required
                />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="confirmPassword">Confirm New Password</label>
                <input
                  type="password"
                  id="confirmPassword"
                  className="form-input"
                  value={passwordForm.confirm}
                  onChange={(e) => setPasswordForm({...passwordForm, confirm: e.target.value})}
                  required
                />
              </div>
            </div>

            <button type="submit" className="btn-primary">Update Password</button>
          </form>
        </div>
      </div>
    </div>
  );
}