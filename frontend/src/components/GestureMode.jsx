import { useState, useRef, useCallback, useEffect } from 'react';
import axios from 'axios';
import './GestureMode.css';

const API = import.meta.env.VITE_API_URL || 'http://localhost:5000';
const CAPTURE_INTERVAL = 300; // ms

export default function GestureMode() {
  /* ── state ── */
  const [streaming, setStreaming] = useState(false);
  const [label, setLabel] = useState(null);
  const [confidence, setConfidence] = useState(0);
  const [sentence, setSentence] = useState([]);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [selectedModel, setSelectedModel] = useState('cnn');

  /* ── refs ── */
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const timerRef = useRef(null);

  /* ── camera lifecycle ── */
  const startCamera = useCallback(async () => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: 640, height: 480 },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setStreaming(true);
    } catch (err) {
      setCameraError('Cannot access webcam. Check permissions.');
      console.error(err);
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setStreaming(false);
    setLabel(null);
    setConfidence(0);
  }, []);

  /* ── capture & predict loop ── */
  useEffect(() => {
    if (!streaming) return;

    const capture = async () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || video.readyState < 2) return;

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0);

      const dataUrl = canvas.toDataURL('image/jpeg', 0.75);

      try {
        const res = await axios.post(`${API}/api/predict?model=${selectedModel}`, { frame: dataUrl });
        setLabel(res.data.label);
        setConfidence(res.data.confidence);
      } catch {
        /* network blip – keep going */
      }
    };

    timerRef.current = setInterval(capture, CAPTURE_INTERVAL);
    return () => {
      clearInterval(timerRef.current);
      timerRef.current = null;
    };
  }, [streaming, selectedModel]);

  /* cleanup on unmount */
  useEffect(() => () => stopCamera(), [stopCamera]);

  /* ── actions ── */
  const addToSentence = () => {
    if (label) setSentence((prev) => [...prev, label]);
  };

  const clearSentence = () => setSentence([]);

  const speakSentence = async () => {
    const text = sentence.join(' ');
    if (!text) return;
    setIsSpeaking(true);
    try {
      const res = await axios.post(`${API}/api/tts`, { text }, { responseType: 'blob' });
      const url = URL.createObjectURL(res.data);
      const audio = new Audio(url);
      audio.onended = () => {
        setIsSpeaking(false);
        URL.revokeObjectURL(url);
      };
      audio.play();
    } catch {
      setIsSpeaking(false);
    }
  };

  /* ── confidence colour ── */
  const confPercent = Math.round(confidence * 100);
  const barColor =
    confPercent >= 75 ? 'var(--green)' : confPercent >= 40 ? 'var(--amber)' : 'var(--red)';

  /* ── render ── */
  return (
    <section className="gesture-mode">
      <div className="gesture-grid">
        {/* ──────── Video panel ──────── */}
        <div className="card video-panel">
          <div className="panel-header">
            <span className={`status-dot ${streaming ? 'live' : 'off'}`} />
            <span className="panel-title">
              {streaming ? 'Camera Active' : 'Camera Off'}
            </span>

            {/* ── Model Toggle ── */}
            <div className="model-toggle" id="model-toggle">
              <button
                className={`toggle-btn ${selectedModel === 'svm' ? 'active' : ''}`}
                onClick={() => setSelectedModel('svm')}
              >
                SVM
              </button>
              <button
                className={`toggle-btn ${selectedModel === 'cnn' ? 'active' : ''}`}
                onClick={() => setSelectedModel('cnn')}
              >
                CNN
              </button>
            </div>
          </div>

          <div className="video-wrapper">
            <video ref={videoRef} className="webcam-feed" muted playsInline />
            {!streaming && !cameraError && (
              <div className="video-placeholder">
                <span className="placeholder-icon">Camera</span>
                <p>Initialize the webcam to begin recognition.</p>
              </div>
            )}
            {cameraError && (
              <div className="video-placeholder error">
                <span className="placeholder-icon">Error</span>
                <p>{cameraError}</p>
              </div>
            )}
          </div>
          {/* hidden canvas for frame capture */}
          <canvas ref={canvasRef} style={{ display: 'none' }} />

          <div className="cam-controls">
            {!streaming ? (
              <button id="btn-start-camera" className="btn btn-primary" onClick={startCamera}>
                Start Camera
              </button>
            ) : (
              <button id="btn-stop-camera" className="btn btn-danger" onClick={stopCamera}>
                Stop Camera
              </button>
            )}
          </div>
        </div>

        {/* ──────── Results panel ──────── */}
        <div className="results-panel">
          {/* Prediction */}
          <div className="card prediction-card">
            <h3 className="card-title">Prediction Output</h3>
            <div className="predicted-label">{label ?? '--'}</div>

            <div className="confidence-section">
              <div className="confidence-header">
                <span>Classification Confidence</span>
                <span className="confidence-value">{confPercent}%</span>
              </div>
              <div className="progress-track">
                <div
                  className="progress-fill"
                  style={{ width: `${confPercent}%`, background: barColor }}
                />
              </div>
            </div>

            <div className="prediction-actions">
              <button
                id="btn-add-to-sentence"
                className="btn btn-accent"
                disabled={!label}
                onClick={addToSentence}
              >
                Add to Sequence
              </button>
            </div>
          </div>

          {/* Sentence */}
          <div className="card sentence-card">
            <h3 className="card-title">Translation Sequence</h3>

            <div className="sentence-display">
              {sentence.length > 0 ? (
                <p className="sentence-text">{sentence.join(' ')}</p>
              ) : (
                <p className="sentence-empty">Add predicted signs to construct a translation...</p>
              )}
            </div>

            <div className="sentence-actions">
              <button
                id="btn-speak"
                className="btn btn-green"
                disabled={sentence.length === 0 || isSpeaking}
                onClick={speakSentence}
              >
                {isSpeaking ? 'Processing Audio...' : 'Vocalize Text'}
              </button>
              <button
                id="btn-clear-sentence"
                className="btn btn-ghost"
                disabled={sentence.length === 0}
                onClick={clearSentence}
              >
                Clear Results
              </button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
