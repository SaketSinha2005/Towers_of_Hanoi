import { Outlet, NavLink, useNavigate } from 'react-router-dom';
import {
  Brain,
  UserPlus,
  Upload,
  Settings as SettingsIcon,
  LogOut,
  LayoutDashboard,
  ScanEye // Importing a new icon for the brain viewer
} from 'lucide-react';

export default function DashboardLayout({ onLogout }) {
  const navigate = useNavigate();

  const handleLogout = () => {
    // 1. Clear session data from localStorage
    localStorage.removeItem('neurovision_doctor_id');
    localStorage.removeItem('neurovision_doctor_name');

    // 2. Trigger the logout state in App.jsx
    onLogout();

    // 3. Redirect to the landing page
    navigate('/');
  };

  const navItemStyle = (isActive) => ({
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
    padding: '0.75rem 1rem',
    borderRadius: '6px',
    color: isActive ? 'white' : '#94a3b8',
    backgroundColor: isActive ? 'rgba(255, 255, 255, 0.1)' : 'transparent',
    transition: 'all 0.2s',
    marginBottom: '0.5rem',
    textDecoration: 'none' // Ensures links don't have underlines
  });

  return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: '#f1f5f9' }}>
      {/* Sidebar */}
      <aside style={{ width: '250px', backgroundColor: '#0f172a', color: 'white', display: 'flex', flexDirection: 'column' }}>
        <div style={{ padding: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.75rem', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
          <Brain size={28} color="#3b82f6" />
          <span style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>NeuroVision AI</span>
        </div>

        <nav style={{ padding: '1.5rem 1rem', flex: 1 }}>
          <NavLink to="/dashboard" end style={({ isActive }) => navItemStyle(isActive)}>
            <LayoutDashboard size={20} /> Dashboard
          </NavLink>
          <NavLink to="/dashboard/create-patient" style={({ isActive }) => navItemStyle(isActive)}>
            <UserPlus size={20} /> Create Patient
          </NavLink>
          <NavLink to="/dashboard/upload" style={({ isActive }) => navItemStyle(isActive)}>
            <Upload size={20} /> Upload MRI Scan
          </NavLink>


        </nav>

        <div style={{ padding: '1rem', borderTop: '1px solid rgba(255,255,255,0.1)' }}>
          <NavLink to="/dashboard/settings" style={({ isActive }) => navItemStyle(isActive)}>
            <SettingsIcon size={20} /> Settings
          </NavLink>
          <button
            onClick={handleLogout}
            style={{
              ...navItemStyle(false),
              width: '100%',
              border: '1px solid #475569',
              marginTop: '0.5rem',
              cursor: 'pointer',
              background: 'none'
            }}
          >
            <LogOut size={20} /> Log Out
          </button>
        </div>
      </aside>

      {/* Main Content Area */}
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column', height: '100vh', overflowY: 'auto' }}>
        <div style={{ padding: '2rem' }}>
          <Outlet />
        </div>
      </main>
    </div>
  );
}