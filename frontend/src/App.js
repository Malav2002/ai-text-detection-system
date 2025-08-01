import React, { useState, useRef } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8001';

function App() {
  const [text, setText] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('text');
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleTextPredict = async () => {
    if (!text.trim()) {
      setError('Please enter some text to analyze');
      return;
    }

    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/predict`, {
        text: text,
        model_type: 'bert'
      });

      setPrediction(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to analyze text');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (file) => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setPrediction(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_type', 'bert');

    try {
      const response = await axios.post(`${API_BASE_URL}/predict/file`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setPrediction(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to analyze file');
    } finally {
      setLoading(false);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileInputChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileUpload(e.target.files[0]);
    }
  };

  const resetAll = () => {
    setText('');
    setPrediction(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const getPredictionColor = (prediction) => {
    if (prediction === 'AI-generated') {
      return '#ef4444'; // red
    } else {
      return '#10b981'; // green
    }
  };

  const sampleTexts = [
    {
      label: "Human-like",
      text: "I've been thinking about this topic for a while now, and I have to say, it's really quite fascinating. From my personal experience, I've noticed that..."
    },
    {
      label: "AI-like", 
      text: "The implementation demonstrates optimal algorithmic efficiency and performance metrics across multiple computational paradigms, ensuring scalable solutions for enterprise applications."
    }
  ];

  return (
    <div className="App">
      <header className="app-header">
        <div className="container">
          <h1>🤖 AI Text Detection System</h1>
          <p>Detect whether text is AI-generated or human-written using advanced BERT models</p>
        </div>
      </header>

      <main className="container">
        <div className="tabs">
          <button 
            className={`tab ${activeTab === 'text' ? 'active' : ''}`}
            onClick={() => setActiveTab('text')}
          >
            📝 Text Input
          </button>
          <button 
            className={`tab ${activeTab === 'file' ? 'active' : ''}`}
            onClick={() => setActiveTab('file')}
          >
            📄 File Upload
          </button>
        </div>

        {activeTab === 'text' && (
          <div className="tab-content">
            <div className="input-section">
              <div className="sample-texts">
                <h3>Try Sample Texts:</h3>
                <div className="sample-buttons">
                  {sampleTexts.map((sample, index) => (
                    <button
                      key={index}
                      className="sample-button"
                      onClick={() => setText(sample.text)}
                    >
                      {sample.label}
                    </button>
                  ))}
                </div>
              </div>

              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Enter or paste text here to analyze..."
                className="text-input"
                rows={8}
              />
              
              <div className="button-group">
                <button
                  onClick={handleTextPredict}
                  disabled={loading || !text.trim()}
                  className="predict-button"
                >
                  {loading ? '🔍 Analyzing...' : '🎯 Analyze Text'}
                </button>
                
                <button
                  onClick={resetAll}
                  className="reset-button"
                >
                  🔄 Clear
                </button>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'file' && (
          <div className="tab-content">
            <div 
              className={`file-upload-area ${dragActive ? 'drag-active' : ''}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <div className="upload-content">
                <div className="upload-icon">📄</div>
                <h3>Upload File for Analysis</h3>
                <p>Drag & drop files here, or click to select</p>
                <p className="file-types">Supports: TXT, PDF, DOCX, JPG, PNG</p>
                
                <input
                  ref={fileInputRef}
                  type="file"
                  onChange={handleFileInputChange}
                  accept=".txt,.pdf,.docx,.jpg,.jpeg,.png"
                  className="file-input"
                  id="file-input"
                />
                
                <label htmlFor="file-input" className="file-upload-button">
                  Choose File
                </label>
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="error-message">
            <span className="error-icon">⚠️</span>
            {error}
          </div>
        )}

        {prediction && (
          <div className="result-section">
            <div className="result-card">
              <h2>📊 Analysis Result</h2>
              
              <div className="prediction-result">
                <div 
                  className="prediction-label"
                  style={{ color: getPredictionColor(prediction.prediction) }}
                >
                  {prediction.prediction === 'AI-generated' ? '🤖' : '👤'} {prediction.prediction}
                </div>
                
                <div className="confidence-meter">
                  <div className="confidence-label">
                    Confidence: {(prediction.confidence * 100).toFixed(1)}%
                  </div>
                  <div className="confidence-bar">
                    <div 
                      className="confidence-fill"
                      style={{ 
                        width: `${prediction.confidence * 100}%`,
                        backgroundColor: getPredictionColor(prediction.prediction)
                      }}
                    ></div>
                  </div>
                </div>

                <div className="result-details">
                  <div className="detail-item">
                    <span className="detail-label">Model Used:</span>
                    <span className="detail-value">BERT-base</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Processing Time:</span>
                    <span className="detail-value">{(prediction.processing_time * 1000).toFixed(0)}ms</span>
                  </div>
                </div>
              </div>

              <div className="explanation">
                <h3>💡 What this means:</h3>
                {prediction.prediction === 'AI-generated' ? (
                  <p>This text appears to be generated by an AI model. It may contain patterns typical of automated content generation, such as formal language, repetitive structures, or lack of personal context.</p>
                ) : (
                  <p>This text appears to be written by a human. It likely contains natural language patterns, personal experiences, or informal expressions that are characteristic of human writing.</p>
                )}
              </div>
            </div>
          </div>
        )}

        <div className="info-section">
          <h2>🔬 How it works</h2>
          <div className="info-grid">
            <div className="info-card">
              <h3>🧠 BERT Model</h3>
              <p>Uses a fine-tuned BERT transformer model trained on thousands of AI and human text samples</p>
            </div>
            <div className="info-card">
              <h3>📈 High Accuracy</h3>
              <p>Achieves high precision in distinguishing between AI-generated and human-written content</p>
            </div>
            <div className="info-card">
              <h3>⚡ Fast Analysis</h3>
              <p>Get results in milliseconds with real-time text analysis and processing</p>
            </div>
          </div>
        </div>
      </main>

      <footer className="app-footer">
        <div className="container">
          <p>Built with ❤️ using React, FastAPI, and BERT • Group 16 NLP Project</p>
        </div>
      </footer>
    </div>
  );
}

export default App;