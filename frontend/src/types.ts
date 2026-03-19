export interface StudentProfile {
  age: number;
  hours_work: number;
  social_media_use: number;
  rent: number;
  friends_count: number;
  highest_speed: number;
  dates: number;
  standard_drinks: number;
  countries: number;
  semesters: number;
  commute: number;
  data_interest: number;
  mark_goal: number;
  hours_studying: number;
  gender: string;
  relationship_status: string;
  drug_use_ans: string;
  student_type: string;
  mainstream_advanced: string;
  lecture_mode: string;
  study_type: string;
  learner_style: string;
}

export interface Prediction {
  model_name: string;
  category: string;
}

export interface PredictResponse {
  predictions: Prediction[];
}
