import { useState, useRef, useCallback } from 'react';
import axios from 'axios';
import './TextToSign.css';

const API = import.meta.env.VITE_API_URL || 'http://localhost:5000';

export default function TextToSign() {
  /* ── state ── */
  const [text, setText] = useState('');
  const [clips, setClips] = useState([]);
  const [currentIdx, setCurrentIdx] = useState(-1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [finished, setFinished] = useState(false);

  const videoRef = useRef(null);

  /* ── fetch clips ── */
  const handleSubmit = useCallback(
    async (e) => {
      e.preventDefault();
      if (!text.trim()) return;

      setLoading(true);
      setError(null);
      setClips([]);
      setCurrentIdx(-1);
      setFinished(false);

      try {
        const res = await axios.post(`${API}/api/text-to-sign`, { text });
        const c = res.data.clips || [];
        if (c.length === 0) {
          setError('No matching sign videos found.');
          setLoading(false);
          return;
        }
        setClips(c);
        setCurrentIdx(0);
      } catch {
        setError('Failed to fetch sign videos.');
      } finally {
        setLoading(false);
      }
    },
    [text],
  );

  /* ── sequential playback ── */
  const handleVideoEnded = useCallback(() => {
    const nextIdx = currentIdx + 1;
    if (nextIdx < clips.length) {
      setCurrentIdx(nextIdx);
    } else {
      setFinished(true);
    }
  }, [currentIdx, clips.length]);

  /* Derive display label from filename (strip .mp4) */
  const currentClip = clips[currentIdx] || null;
  const currentLabel = currentClip
    ? currentClip.replace(/\.mp4$/i, '')
    : null;

  /* ── render ── */
  return (
    <section className="text-to-sign">
      <div className="t2s-grid">
        {/* ──────── Input panel ──────── */}
        <div className="card input-panel">
          <h3 className="card-title">Text Input</h3>
          <form onSubmit={handleSubmit} className="t2s-form">
            <textarea
              id="t2s-input"
              className="t2s-textarea"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter English text for translation..."
              rows={3}
            />
            <button
              id="btn-translate"
              type="submit"
              className="btn btn-primary"
              disabled={loading || !text.trim()}
            >
              {loading ? 'Processing...' : 'Translate to Sign Language'}
            </button>
          </form>

          {error && <p className="t2s-error">Error: {error}</p>}

          {/* Clip timeline */}
          {clips.length > 0 && (
            <div className="clip-timeline">
              <h4 className="timeline-title">Animation Sequence</h4>
              <div className="timeline-chips">
                {clips.map((c, i) => (
                  <span
                    key={i}
                    className={`chip${
                      i === currentIdx
                        ? ' chip-active'
                        : i < currentIdx || finished
                        ? ' chip-done'
                        : ''
                    }`}
                  >
                    {c.replace(/\.mp4$/i, '')}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* ──────── Video panel ──────── */}
        <div className="card video-panel-t2s">
          <div className="panel-header">
            <span className={`status-dot ${currentIdx >= 0 && !finished ? 'live' : 'off'}`} />
            <span className="panel-title">
              {finished
                ? 'Visualization Complete'
                : currentIdx >= 0
                ? 'Active Visualization'
                : 'Awaiting Input'}
            </span>
          </div>

          <div className="video-wrapper-t2s">
            {currentClip ? (
              <video
                ref={videoRef}
                key={currentClip}          /* remount on clip change */
                className="sign-video"
                autoPlay
                onEnded={handleVideoEnded}
                src={`${API}/videos/${encodeURIComponent(currentClip)}`}
              />
            ) : (
              <div className="video-placeholder-t2s">
                <span className="placeholder-icon">Animation View</span>
                <p>Input text and select Translate to generate visualization.</p>
              </div>
            )}
          </div>

          {currentLabel && (
            <div className="now-playing">
              <span className="now-label">Current Sign:</span>
              <span className="now-value">{currentLabel}</span>
              <span className="now-counter">
                Segment {currentIdx + 1} of {clips.length}
              </span>
            </div>
          )}

          {finished && clips.length > 0 && (
            <button
              id="btn-replay"
              className="btn btn-accent replay-btn"
              onClick={() => {
                setFinished(false);
                setCurrentIdx(0);
              }}
            >
              Replay Visualization
            </button>
          )}
        </div>
      </div>
    </section>
  );
}
