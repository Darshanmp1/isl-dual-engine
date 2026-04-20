import { useState } from 'react';
import GestureMode from './components/GestureMode';
import TextToSign from './components/TextToSign';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('gesture');

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="app-header">
        <h1>Smart ISL Translator</h1>
      </header>

      {/* ── Tabs ── */}
      <nav className="tab-bar" role="tablist" aria-label="Translator mode">
        <button
          id="tab-gesture"
          role="tab"
          aria-selected={activeTab === 'gesture'}
          aria-controls="panel-gesture"
          className={`tab-btn${activeTab === 'gesture' ? ' active' : ''}`}
          onClick={() => setActiveTab('gesture')}
        >
          Sign → Text
        </button>
        <button
          id="tab-text2sign"
          role="tab"
          aria-selected={activeTab === 'text2sign'}
          aria-controls="panel-text2sign"
          className={`tab-btn${activeTab === 'text2sign' ? ' active' : ''}`}
          onClick={() => setActiveTab('text2sign')}
        >
          Text → Sign
        </button>
      </nav>

      {/* ── Panels ── */}
      <main className="tab-content">
        {activeTab === 'gesture' && (
          <div id="panel-gesture" role="tabpanel" aria-labelledby="tab-gesture">
            <GestureMode />
          </div>
        )}
        {activeTab === 'text2sign' && (
          <div id="panel-text2sign" role="tabpanel" aria-labelledby="tab-text2sign">
            <TextToSign />
          </div>
        )}
      </main>

      {/* ── Footer ── */}
      <footer className="app-footer">
        Smart AI-Powered Indian Sign Language Translator &middot; Bridging the communication gap
      </footer>
    </div>
  );
}

export default App;
