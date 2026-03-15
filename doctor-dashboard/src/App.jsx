import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { useState, useEffect } from 'react';

// Pages
import Home from './pages/Home';
import Login from './pages/Login';
import Signup from './pages/Signup';
import DashboardLayout from './layouts/DashboardLayout';
import Dashboard from './pages/Dashboard';
import CreatePatient from './pages/CreatePatient';
import UploadMRI from './pages/UploadMRI';
import Analysis from './pages/Analysis';
import PatientHistory from './pages/PatientHistory';
import Settings from './pages/Settings'; // Import the new viewer page

function App() {
  // Initialize state based on whether a doctor ID exists in storage
  const [isAuthenticated, setIsAuthenticated] = useState(
    !!localStorage.getItem('neurovision_doctor_id')
  );

  const login = () => setIsAuthenticated(true);
  const logout = () => setIsAuthenticated(false);

  return (
    <Router>
      <Routes>
        {/* Public Routes */}
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<Login onLogin={login} />} />
        <Route path="/signup" element={<Signup onSignup={login} />} />

        {/* Protected Routes inside DashboardLayout */}
        <Route
          path="/dashboard"
          element={isAuthenticated ? <DashboardLayout onLogout={logout} /> : <Navigate to="/login" />}
        >
          <Route index element={<Dashboard />} />
          <Route path="create-patient" element={<CreatePatient />} />
          <Route path="upload" element={<UploadMRI />} />

          <Route path="analysis/:patientId" element={<Analysis />} />
          <Route path="patient/:patientId" element={<PatientHistory />} />
          <Route path="settings" element={<Settings />} />
        </Route>

        {/* Fallback */}
        <Route path="*" element={<Navigate to="/" />} />
      </Routes>
    </Router>
  );
}

export default App;