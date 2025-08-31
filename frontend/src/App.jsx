import React, { useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE ?? ''           
const OWNER_NAME = import.meta.env.VITE_APP_OWNER ?? 'Aymen Ouerghemmi'
const CONTACT_EMAIL = import.meta.env.VITE_APP_CONTACT_EMAIL ?? 'werghemiaymen@gmail.com'

export default function App() {
  const [file, setFile] = useState(null)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const year = new Date().getFullYear()

  const onChange = (e) => {
    setFile(e.target.files?.[0] ?? null)
    setResult(null)
    setError(null)
  }

  const onSubmit = async () => {
    if (!file) return
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      const form = new FormData()
      form.append('file', file)
      const r = await fetch(`${API_BASE}/infer`, { method: 'POST', body: form })
      if (!r.ok) throw new Error(`Request failed (${r.status})`)
      const data = await r.json()
      setResult(data)
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="container">
      <header>
        <h1>Medical Scanner + RAG</h1>
        <span className="pill">Demo • Do not use clinically</span>
      </header>

      <div className="card">
        <h3>1) Upload CT image (DICOM .dcm or PNG/JPG)</h3>
        <input type="file" accept=".dcm,.dicom,.png,.jpg,.jpeg" onChange={onChange} />
        <div style={{marginTop:12}}>
          <button className="btn" onClick={onSubmit} disabled={!file || busy}>
            {busy ? 'Analyzing…' : 'Run analysis'}
          </button>
        </div>
        {busy && (
          <div style={{marginTop:12}} className="progress">
            <div style={{width:'80%'}}></div>
          </div>
        )}
        {error && <p style={{color:'#b91c1c'}}>Error: {error}</p>}
      </div>

      {result && (
        <div className="row">
          <div className="card">
            <h3>Prediction</h3>
            <p className="muted">Top class: <strong>{result.top_label}</strong></p>
            <div style={{display:'grid', gap:8}}>
              {Object.entries(result.predictions).map(([label, p]) => (
                <div key={label}>
                  <div style={{display:'flex', justifyContent:'space-between'}}>
                    <span className="badge">{label}</span>
                    <span>{(p*100).toFixed(1)}%</span>
                  </div>
                  <div className="bar"><div style={{width: (p*100)+'%'}}></div></div>
                </div>
              ))}
            </div>
            <div style={{marginTop:12}} className="mono">
              {JSON.stringify(result.predictions, null, 2)}
            </div>
          </div>
          <div className="card">
            <h3>Grad-CAM</h3>
            <p className="muted">Model attention overlay (rough localization).</p>
            <img src={`${API_BASE}${result.gradcam_url}`} style={{width:'100%', borderRadius:12}} alt="gradcam" />
          </div>
        </div>
      )}

      {result?.rag?.length > 0 && (
        <div className="card">
          <h3>Knowledge (RAG)</h3>
          <ol>
            {result.rag.map((r) => (
              <li key={r.doc} style={{marginBottom:12}}>
                <div style={{display:'flex',gap:8, alignItems:'center'}}>
                  <span className="badge">score {(r.score).toFixed(2)}</span>
                  <strong>{r.doc}</strong>
                </div>
                <p className="muted">{r.snippet}</p>
              </li>
            ))}
          </ol>
        </div>
      )}

      <div className="footer">
        Built for demo/educational purposes only. Not medical advice.
        <div style={{marginBottom:6}}>
          © {year} {OWNER_NAME}. Tous droits réservés.
        </div>
        <div>
          Problème ou question ? Contact :{' '}
          <a href={`mailto:${CONTACT_EMAIL}`}>{CONTACT_EMAIL}</a>
        </div>
      </div>
    </div>
  )
}
