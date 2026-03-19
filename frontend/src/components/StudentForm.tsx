import type { StudentProfile } from '../types';
import './StudentForm.css';

interface Props {
  profile: StudentProfile;
  onInputChange: (field: keyof StudentProfile, value: string) => void;
  onLoadRandom: () => void;
  onLoadTypical: () => void;
  onPredict: () => void;
  loading: boolean;
  profileLoading: 'random' | 'typical' | null;
}

interface NumericField {
  key: keyof StudentProfile;
  label: string;
  min: number;
  max: number;
  step: number;
}

interface SelectField {
  key: keyof StudentProfile;
  label: string;
  options: string[];
}

const NUMERIC_FIELDS: NumericField[] = [
  { key: 'age',              label: 'Age',                   min: 16,  max: 50,   step: 1   },
  { key: 'hours_work',       label: 'Work Hrs / Week',       min: 0,   max: 80,   step: 0.5 },
  { key: 'social_media_use', label: 'Social Media Hrs / Day',min: 0,   max: 24,   step: 0.5 },
  { key: 'rent',             label: 'Weekly Rent ($)',       min: 0,   max: 5000, step: 1   },
  { key: 'friends_count',    label: 'Close Friends',         min: 0,   max: 100,  step: 1   },
  { key: 'highest_speed',    label: 'Internet Speed (Mbps)', min: 0,   max: 1000, step: 1   },
  { key: 'dates',            label: 'Dates / Month',         min: 0,   max: 30,   step: 1   },
  { key: 'standard_drinks',  label: 'Std Drinks / Week',     min: 0,   max: 50,   step: 1   },
  { key: 'countries',        label: 'Countries Visited',     min: 0,   max: 200,  step: 1   },
  { key: 'semesters',        label: 'Semesters Done',        min: 0,   max: 20,   step: 1   },
  { key: 'commute',          label: 'Commute (min)',         min: 0,   max: 300,  step: 1   },
  { key: 'data_interest',    label: 'Data Interest (1–10)',  min: 1,   max: 10,   step: 1   },
  { key: 'mark_goal',        label: 'Mark Goal (%)',         min: 0,   max: 100,  step: 1   },
  { key: 'hours_studying',   label: 'Study Hrs / Week',      min: 0,   max: 100,  step: 0.5 },
];

const SELECT_FIELDS: SelectField[] = [
  {
    key: 'gender',
    label: 'Gender',
    options: ['Male', 'Female', 'Other'],
  },
  {
    key: 'relationship_status',
    label: 'Relationship Status',
    options: ['Single', 'In a relationship', 'Its complicated', 'Married'],
  },
  {
    key: 'drug_use_ans',
    label: 'Drug Use',
    options: ['Yes', 'No', 'Prefer not to say'],
  },
  {
    key: 'student_type',
    label: 'Student Type',
    options: ['Domestic', 'International'],
  },
  {
    key: 'mainstream_advanced',
    label: 'Course',
    options: ['DATA1001', 'DATA1901'],
  },
  {
    key: 'lecture_mode',
    label: 'Lecture Mode',
    options: ['Live in the Lecture Theatre', 'Live Online', 'Recorded', 'Other'],
  },
  {
    key: 'study_type',
    label: 'Study Pattern',
    options: [
      'I work steadily all semester',
      'It changes depending on the subject',
      'Other',
    ],
  },
  {
    key: 'learner_style',
    label: 'Learner Style',
    options: ['Style 1', 'Style 2', 'Style 3'],
  },
];

function StudentForm({ profile, onInputChange, onLoadRandom, onLoadTypical, onPredict, loading, profileLoading }: Props) {
  return (
    <form className="student-form" noValidate onSubmit={(e) => { e.preventDefault(); onPredict(); }}>
      <h2>Student Profile</h2>

      <div className="form-section-header">
        <h3><span className="section-icon">📊</span> Numeric Features</h3>
        <p>Enter student habits and circumstances</p>
      </div>

      <div className="form-grid">
        {NUMERIC_FIELDS.map(({ key, label, min, max, step }) => (
          <div className="form-group" key={key}>
            <label htmlFor={key}>{label}</label>
            <input
              id={key}
              type="number"
              value={profile[key] as number}
              onChange={(e) => onInputChange(key, e.target.value)}
              min={min}
              max={max}
              step={step}
            />
          </div>
        ))}
      </div>

      <div className="form-section-header">
        <h3><span className="section-icon">🏷️</span> Categorical Features</h3>
        <p>Select from the available options</p>
      </div>

      <div className="form-grid-select">
        {SELECT_FIELDS.map(({ key, label, options }) => (
          <div className="form-group" key={key}>
            <label htmlFor={key}>{label}</label>
            <select
              id={key}
              value={profile[key] as string}
              onChange={(e) => onInputChange(key, e.target.value)}
            >
              {options.map((opt) => (
                <option key={opt} value={opt}>{opt}</option>
              ))}
            </select>
          </div>
        ))}
      </div>

      <div className="form-actions">
        <button
          type="button"
          className="btn btn-ghost"
          onClick={onLoadRandom}
          disabled={loading || profileLoading !== null}
        >
          {profileLoading === 'random' ? <><span className="btn-spinner btn-spinner--dark" /> Loading…</> : '🎲 Random'}
        </button>
        <button
          type="button"
          className="btn btn-ghost"
          onClick={onLoadTypical}
          disabled={loading || profileLoading !== null}
        >
          {profileLoading === 'typical' ? <><span className="btn-spinner btn-spinner--dark" /> Loading…</> : '📊 Typical'}
        </button>
        <button
          type="submit"
          className="btn btn-primary"
          disabled={loading || profileLoading !== null}
        >
          {loading ? (
            <><span className="btn-spinner" /> Predicting…</>
          ) : (
            '🚀 Predict'
          )}
        </button>
      </div>
    </form>
  );
}

export default StudentForm;
