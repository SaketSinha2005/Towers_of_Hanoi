// Mock local storage integration to keep data during app session

const INITIAL_PATIENTS = [
  { id: 'A6740A34', name: 'Sarah Chen', gender: 'Female', age: 45, phone: '+1 555-0123', height: '165 cm', weight: '60 kg', lastMriDate: 'May 18, 2024', status: 'Analysis Complete' },
  { id: 'A5340A0B', name: 'Michael Rodriguez', gender: 'Male', age: 52, phone: '+1 555-0124', height: '180 cm', weight: '82 kg', lastMriDate: 'May 17, 2024', status: 'Processing' },
  { id: '5D8A5A24', name: 'David Kim', gender: 'Male', age: 38, phone: '+1 555-0125', height: '175 cm', weight: '75 kg', lastMriDate: 'May 17, 2024', status: 'Requires Review' },
  { id: 'EBA3CA00', name: 'Aisha Khan', gender: 'Female', age: 61, phone: '+1 555-0126', height: '160 cm', weight: '58 kg', lastMriDate: 'May 17, 2024', status: 'Analysis Complete' },
];

const INITIAL_CHECKUPS = {
  'A6740A34': [
    {
      id: 'chk_1',
      date: 'May 18, 2024',
      scans: ['mri_t1.dcm', 'mri_t2.dcm', 'mri_flair.dcm'],
      analysis: 'No acute intracranial pathology detected. Ventricles and sulci are prominent, within normal limits for age. No midline shift or mass effect.'
    },
    {
      id: 'chk_2',
      date: 'Jan 10, 2024',
      scans: ['mri_baseline.dcm'],
      analysis: 'Baseline scan. Normal anatomical structures.'
    }
  ],
  'A5340A0B': [
    {
      id: 'chk_3',
      date: 'May 17, 2024',
      scans: ['mri_t1.dcm', 'mri_t2.dcm'],
      analysis: 'Processing...'
    }
  ]
};

class DataStore {
  constructor() {
    this.patients = [...INITIAL_PATIENTS];
    this.checkups = { ...INITIAL_CHECKUPS };
    this.doctor = {
        name: 'Dr. Jane Smith',
        email: 'jsmith@neurovision.ai',
        hospital: 'Central Hospital'
    };
  }

  getPatients() {
    return this.patients;
  }

  getPatient(id) {
    return this.patients.find(p => p.id === id);
  }

  addPatient(patient) {
    const newPatient = {
      ...patient,
      id: Math.random().toString(36).substring(2, 10).toUpperCase(),
      status: 'No Scans Yet'
    };
    this.patients.unshift(newPatient); // Add to beginning
    return newPatient;
  }

  getCheckups(patientId) {
    return this.checkups[patientId] || [];
  }

  addCheckup(patientId, scans, analysis) {
    if (!this.checkups[patientId]) {
      this.checkups[patientId] = [];
    }
    const checkup = {
      id: 'chk_' + Math.random().toString(36).substring(2, 8),
      date: new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }),
      scans,
      analysis: analysis || 'Analysis Complete. AI reveals standard structural appearance.'
    };
    this.checkups[patientId].unshift(checkup);
    
    // Update patient status
    const ptIndex = this.patients.findIndex(p => p.id === patientId);
    if(ptIndex >= 0) {
        this.patients[ptIndex].status = 'Analysis Complete';
        this.patients[ptIndex].lastMriDate = checkup.date;
    }
    
    return checkup;
  }

  getDoctor() {
    return this.doctor;
  }

  updateDoctor(data) {
    this.doctor = { ...this.doctor, ...data };
    return this.doctor;
  }
}

// Singleton instance
export const dataStore = new DataStore();
