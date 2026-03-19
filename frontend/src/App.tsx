import './App.css';
import StressPredictor from './components/StressPredictor';

function App() {
  return (
    <div className="app">
      <header className="app-header">
        <div className="header-badge">Machine Learning</div>
        <h1>Student Stress Predictor</h1>
        <p>Analyse student lifestyle profiles and their stress.</p>
      </header>
      <main>
        <StressPredictor />
      </main>
    </div>
  );
}

export default App;
