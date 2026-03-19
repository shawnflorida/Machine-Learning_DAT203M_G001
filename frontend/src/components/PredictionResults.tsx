import type { JSX } from 'react';
import './PredictionResults.css';
import type { Prediction, StudentProfile } from '../types';

interface Props {
  predictions: Prediction[];
  profile: StudentProfile;
}

type StressLevel = 'low' | 'average' | 'high' | 'default';

function getLevel(category: string): StressLevel {
  if (category.includes('Low')) return 'low';
  if (category.includes('Average')) return 'average';
  if (category.includes('High')) return 'high';
  return 'default';
}

const LEVEL_EMOJI: Record<StressLevel, string> = {
  low: '😊',
  average: '😐',
  high: '😟',
  default: '😐',
};

function CategoryBadge({ category, level }: { category: string; level: StressLevel }): JSX.Element {
  return (
    <div className={`category-display level-${level}`}>
      <span className="cat-emoji">{LEVEL_EMOJI[level]}</span>
      <span className={`category-badge level-${level}`}>{category}</span>
    </div>
  );
}

function PredictionResults({ predictions, profile }: Props) {
  return (
    <div className="prediction-results">
      <h2>Prediction Results</h2>

      <div className="results-grid">
        {predictions.map((pred, idx) => {
          const level = getLevel(pred.category);
          return (
            <div key={idx} className={`prediction-card level-${level}`}>
              <div className="card-header">
                <h3>{pred.model_name}</h3>
              </div>
              <div className="card-body">
                <span className="field-label">Predicted Stress Level</span>
                <CategoryBadge category={pred.category} level={level} />
              </div>
            </div>
          );
        })}
      </div>

      <div className="profile-summary">
        <h3>Profile Snapshot</h3>
        <div className="summary-grid">
          {[
            { label: 'Work',          value: `${parseFloat(Number(profile.hours_work).toFixed(1))} hrs/week` },
            { label: 'Social Media',  value: `${parseFloat(Number(profile.social_media_use).toFixed(1))} hrs/day` },
            { label: 'Study',         value: `${Math.round(profile.hours_studying)} hrs/week` },
            { label: 'Friends',       value: `${Math.round(profile.friends_count)} close` },
            { label: 'Relationship',  value: profile.relationship_status },
            { label: 'Rent',          value: `$${Math.round(profile.rent)}/week` },
          ].map(({ label, value }) => (
            <div className="summary-item" key={label}>
              <span className="s-label">{label}</span>
              <span className="s-value">{value}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="interpretation">
        <h3>💡 Interpretation</h3>
        <ul>
          <li><strong className="low-text">Low Stress:</strong> Student is managing well with a healthy balance.</li>
          <li><strong className="avg-text">Average Stress:</strong> Moderate challenges — normal for university.</li>
          <li><strong className="high-text">High Stress:</strong> May benefit from support or time management strategies.</li>
        </ul>
        <p>Models use lifestyle features as input. Stress is subjective — these are data-driven estimates.</p>
      </div>
    </div>
  );
}

export default PredictionResults;
