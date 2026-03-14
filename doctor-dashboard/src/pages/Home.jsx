import { Link } from 'react-router-dom';
import { Brain, Activity, FileText } from 'lucide-react';

export default function Home() {
  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <header style={{ padding: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', backgroundColor: '#0f172a', color: 'white' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Brain size={32} color="#3b82f6" />
          <span style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>NeuroVision AI</span>
        </div>
        <div style={{ display: 'flex', gap: '1rem' }}>
          <Link to="/login" className="btn-outline" style={{ color: 'white', borderColor: 'white' }}>Login</Link>
          <Link to="/signup" className="btn-primary">Sign Up</Link>
        </div>
      </header>

      <main style={{ flex: 1, backgroundColor: '#0f172a', color: 'white', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '4rem 2rem' }}>
        <h1 style={{ fontSize: '3.5rem', textAlign: 'center', marginBottom: '1rem', background: 'linear-gradient(to right, #3b82f6, #8b5cf6)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
          [AI-Powered MRI Analytics]
        </h1>
        <p style={{ fontSize: '1.25rem', color: '#94a3b8', marginBottom: '4rem' }}>[Secure Doctor Access Only.]</p>

        <h2 style={{ fontSize: '2rem', marginBottom: '3rem' }}>ABOUT US</h2>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem', width: '100%', maxWidth: '1200px' }}>
          {/* Feature 1 */}
          <div style={{ background: 'rgba(255,255,255,0.05)', padding: '2rem', borderRadius: '16px', backdropFilter: 'blur(10px)', textAlign: 'center' }}>
            <div style={{ backgroundColor: 'rgba(59, 130, 246, 0.2)', width: '80px', height: '80px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 1.5rem auto' }}>
              <Brain size={40} color="#60a5fa" />
            </div>
            <h3 style={{ fontSize: '1.25rem', marginBottom: '1rem' }}>Automated Segmentation.</h3>
            <p style={{ color: '#cbd5e1', lineHeight: '1.6' }}>Advanced AI models (e.g., U-Net) automate the precise segmentation of brain MRIs for tumor detection and delineation.</p>
          </div>
          
          {/* Feature 2 */}
          <div style={{ background: 'rgba(255,255,255,0.05)', padding: '2rem', borderRadius: '16px', backdropFilter: 'blur(10px)', textAlign: 'center' }}>
            <div style={{ backgroundColor: 'rgba(59, 130, 246, 0.2)', width: '80px', height: '80px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 1.5rem auto' }}>
              <Activity size={40} color="#60a5fa" />
            </div>
            <h3 style={{ fontSize: '1.25rem', marginBottom: '1rem' }}>Volumetric Analytics.</h3>
            <p style={{ color: '#cbd5e1', lineHeight: '1.6' }}>Tracks tumor volume and growth rate over time with predictive modeling, enabling longitudinal tracking of patient progress.</p>
          </div>

          {/* Feature 3 */}
          <div style={{ background: 'rgba(255,255,255,0.05)', padding: '2rem', borderRadius: '16px', backdropFilter: 'blur(10px)', textAlign: 'center' }}>
            <div style={{ backgroundColor: 'rgba(59, 130, 246, 0.2)', width: '80px', height: '80px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 1.5rem auto' }}>
              <FileText size={40} color="#60a5fa" />
            </div>
            <h3 style={{ fontSize: '1.25rem', marginBottom: '1rem' }}>Diagnostic Decision Support.</h3>
            <p style={{ color: '#cbd5e1', lineHeight: '1.6' }}>Provides quantitative metrics and concise, objective data points to support faster and more accurate clinical decisions.</p>
          </div>
        </div>
      </main>
      
      <footer style={{ backgroundColor: '#020617', color: '#64748b', padding: '1.5rem', textAlign: 'center', fontSize: '0.875rem' }}>
        <p style={{ marginBottom: '0.5rem' }}>Contact Us | Privacy Policy</p>
        <p>Copyright © 2026 All rights reserved.</p>
      </footer>
    </div>
  );
}
