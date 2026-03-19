import { useState } from 'react';
import './StressPredictor.css';
import type { StudentProfile, Prediction } from '../types';
import StudentForm from './StudentForm';
import PredictionResults from './PredictionResults';

const INT_FIELDS = new Set<keyof StudentProfile>([
  'age', 'friends_count', 'dates', 'standard_drinks',
  'countries', 'semesters', 'data_interest', 'mark_goal',
  'highest_speed', 'commute', 'rent',
]);

function roundProfile(data: StudentProfile): StudentProfile {
  const result = { ...data };
  for (const key of Object.keys(result) as Array<keyof StudentProfile>) {
    const val = result[key];
    if (typeof val === 'number') {
      (result as Record<string, unknown>)[key] = INT_FIELDS.has(key)
        ? Math.round(val)
        : parseFloat(val.toFixed(1));
    }
  }
  return result;
}

function StressPredictor() {
  const [profile, setProfile] = useState<StudentProfile | null>(null);
  const [loading, setLoading] = useState(false);
  const [profileLoading, setProfileLoading] = useState<'random' | 'typical' | null>(null);
  const [predictions, setPredictions] = useState<Prediction[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchProfile = async (mode: 'random' | 'typical') => {
    setProfileLoading(mode);
    setError(null);
    setPredictions(null);
    try {
      const res = await fetch(`/api/profile/${mode}`);
      if (!res.ok) throw new Error(res.statusText);
      const data: StudentProfile = await res.json();
      setProfile(roundProfile(data));
    } catch (err) {
      setError(`Could not load ${mode} profile: ${(err as Error).message}`);
    } finally {
      setProfileLoading(null);
    }
  };

  const handlePredict = async () => {
    if (!profile) return;
    setLoading(true);
    setError(null);
    setPredictions(null);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(profile),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }

      const data = await response.json();
      setPredictions(data.predictions as Prediction[]);
    } catch (err) {
      setError(`Failed to fetch predictions: ${(err as Error).message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (field: keyof StudentProfile, value: string) => {
    setProfile(prev =>
      prev
        ? {
            ...prev,
            [field]: isNaN(Number(value)) || value === '' ? value : parseFloat(value),
          }
        : prev,
    );
  };

  return (
    <div className="stress-predictor">
      <div className="container">
        {!profile ? (
          <div className="empty-state">
            <div className="empty-icon">🎓</div>
            <h2>Load a Student Profile</h2>
            <p>Generate a profile from the dataset to get started, or use the typical student profile.</p>
            <div className="empty-actions">
              <button
                className="btn btn-primary"
                onClick={() => fetchProfile('random')}
                disabled={profileLoading !== null}
              >
                {profileLoading === 'random' ? <><span className="btn-spinner" /> Loading…</> : '🎲 Random Profile'}
              </button>
              <button
                className="btn btn-secondary"
                onClick={() => fetchProfile('typical')}
                disabled={profileLoading !== null}
              >
                {profileLoading === 'typical' ? <><span className="btn-spinner" /> Loading…</> : '📊 Typical Profile'}
              </button>
            </div>
          </div>
        ) : (
          <div className="form-section">
            <StudentForm
              profile={profile}
              onInputChange={handleInputChange}
              onLoadRandom={() => fetchProfile('random')}
              onLoadTypical={() => fetchProfile('typical')}
              onPredict={handlePredict}
              loading={loading}
              profileLoading={profileLoading}
            />
          </div>
        )}

        {error && (
          <div className="error-message">
            <span className="error-icon">⚠️</span>
            <span>{error}</span>
          </div>
        )}

        {loading && (
          <div className="loading">
            <div className="spinner" />
            <p>Analysing student profile…</p>
          </div>
        )}

        {predictions && !loading && profile && (
          <div className="results-section">
            <PredictionResults predictions={predictions} profile={profile} />
          </div>
        )}
      </div>
    </div>
  );
}

export default StressPredictor;

