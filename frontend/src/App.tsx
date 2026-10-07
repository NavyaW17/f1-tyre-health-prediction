import { useEffect, useState, type FormEvent } from 'react'
import {
  Activity,
  ArrowDownToLine,
  ArrowRight,
  Check,
  CircleHelp,
  Gauge,
  LoaderCircle,
  RefreshCw,
  ShieldCheck,
  TriangleAlert,
  Waves,
} from 'lucide-react'

type FormValues = {
  mileage_km: string
  tyre_pressure_psi: string
  temperature_c: string
  tyre_age_months: string
  average_speed_kmh: string
  road_quality_score: string
}

type Prediction = {
  remaining_life_percentage: number
  health_status: string
  top_factors: string[]
  model_version: string
}

type HistoryItem = {
  id: string
  createdAt: string
  prediction: Prediction
  mileage: number
}

const API_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '')
const INITIAL_VALUES: FormValues = {
  mileage_km: '45000',
  tyre_pressure_psi: '32',
  temperature_c: '35',
  tyre_age_months: '18',
  average_speed_kmh: '70',
  road_quality_score: '6',
}
const FIELDS: { key: keyof FormValues; label: string; unit: string; min: number; max: number; step: number }[] = [
  { key: 'mileage_km', label: 'Vehicle mileage', unit: 'km', min: 0, max: 100000, step: 100 },
  { key: 'tyre_pressure_psi', label: 'Tyre pressure', unit: 'PSI', min: 20, max: 40, step: 0.1 },
  { key: 'temperature_c', label: 'Ambient temperature', unit: '°C', min: 10, max: 70, step: 0.5 },
  { key: 'tyre_age_months', label: 'Tyre age', unit: 'months', min: 0, max: 60, step: 1 },
  { key: 'average_speed_kmh', label: 'Average speed', unit: 'km/h', min: 20, max: 140, step: 1 },
  { key: 'road_quality_score', label: 'Road quality', unit: '/ 10', min: 1, max: 10, step: 0.1 },
]

function statusClass(status: string) {
  return `status-${status.toLowerCase()}`
}

function GaugeMeter({ value, status }: { value: number; status: string }) {
  const radius = 53
  const circumference = 2 * Math.PI * radius
  return (
    <div className="gauge-wrap" aria-label={`${value}% remaining tyre life`}>
      <svg className="gauge-svg" viewBox="0 0 128 128" role="img" aria-hidden="true">
        <circle className="gauge-track" cx="64" cy="64" r={radius} />
        <circle
          className={`gauge-value ${statusClass(status)}`}
          cx="64"
          cy="64"
          r={radius}
          strokeDasharray={`${(value / 100) * circumference} ${circumference}`}
        />
      </svg>
      <div className="gauge-copy">
        <strong>{value}<span>%</span></strong>
        <small>remaining</small>
      </div>
    </div>
  )
}

function TrendChart({ history }: { history: HistoryItem[] }) {
  if (history.length < 2) {
    return (
      <div className="chart-empty">
        <Waves size={18} strokeWidth={1.6} />
        <span>Run another assessment to reveal a trend</span>
      </div>
    )
  }
  const points = history.slice(-8)
  const coords = points.map((item, index) => ({
    x: points.length === 1 ? 48 : 12 + (index / (points.length - 1)) * 76,
    y: 78 - (item.prediction.remaining_life_percentage / 100) * 62,
    item,
  }))
  const line = coords.map(({ x, y }, index) => `${index === 0 ? 'M' : 'L'} ${x} ${y}`).join(' ')
  return (
    <div className="chart-wrap">
      <svg viewBox="0 0 100 88" preserveAspectRatio="none" className="trend-svg" role="img" aria-label="Remaining tyre life across recent assessments">
        {[16, 36, 56, 76].map((y) => <line key={y} x1="8" x2="98" y1={y} y2={y} className="chart-grid" />)}
        <path d={line} className="chart-line" />
        {coords.map(({ x, y, item }) => <circle key={item.id} cx={x} cy={y} r="1.7" className="chart-dot" />)}
      </svg>
      <div className="chart-labels">
        <span>{new Date(points[0].createdAt).toLocaleDateString()}</span>
        <span>{new Date(points[points.length - 1].createdAt).toLocaleDateString()}</span>
      </div>
    </div>
  )
}

function App() {
  const [values, setValues] = useState<FormValues>(INITIAL_VALUES)
  const [result, setResult] = useState<Prediction | null>(null)
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [apiOnline, setApiOnline] = useState<boolean | null>(null)

  useEffect(() => {
    let active = true
    const checkApi = async () => {
      try {
        const response = await fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(4000) })
        if (active) setApiOnline(response.ok)
      } catch {
        if (active) setApiOnline(false)
      }
    }
    void checkApi()
    const timer = window.setInterval(checkApi, 30000)
    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [])

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setLoading(true)
    setError('')
    try {
      const payload = Object.fromEntries(
        Object.entries(values).map(([key, value]) => [key, Number(value)]),
      )
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!response.ok) {
        const body = await response.json().catch(() => ({}))
        throw new Error(body.detail || `Prediction request failed (${response.status})`)
      }
      const prediction = await response.json() as Prediction
      const item: HistoryItem = {
        id: crypto.randomUUID(),
        createdAt: new Date().toISOString(),
        prediction,
        mileage: Number(values.mileage_km),
      }
      setResult(prediction)
      setHistory((current) => [...current, item].slice(-20))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Unable to connect to the prediction service.')
    } finally {
      setLoading(false)
    }
  }

  const exportHistory = () => {
    if (!history.length) return
    const rows = [
      ['assessed_at', 'mileage_km', 'remaining_life_percentage', 'health_status', 'model_version'],
      ...history.map(({ createdAt, mileage, prediction }) => [
        createdAt,
        String(mileage),
        String(prediction.remaining_life_percentage),
        prediction.health_status,
        prediction.model_version,
      ]),
    ]
    const blob = new Blob([rows.map((row) => row.map((cell) => `"${cell.replaceAll('"', '""')}"`).join(',')).join('\n')], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = 'tyre-health-history.csv'
    link.click()
    URL.revokeObjectURL(url)
  }

  const latest = history.at(-1)

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#dashboard" aria-label="Tyre Health dashboard home">
          <span className="brand-mark"><Gauge size={21} strokeWidth={1.8} /></span>
          <span className="brand-name">TREAD<span>/</span>LINE</span>
        </a>
        <div className="topbar-meta">
          <span className={`service-state ${apiOnline === true ? 'is-online' : apiOnline === false ? 'is-offline' : ''}`}>
            <span className="service-dot" />
            {apiOnline === null ? 'Connecting' : apiOnline ? 'Model service online' : 'Service unavailable'}
          </span>
          <span className="version-label">PREDICTOR <b>v1.0</b></span>
        </div>
      </header>

      <section className="intro" id="dashboard">
        <div>
          <div className="eyebrow"><span className="eyebrow-line" /> TYRE CONDITION MONITOR</div>
          <h1>Know what’s<br /><em>under your wheels.</em></h1>
          <p className="intro-copy">Estimate remaining tyre life from real-world driving conditions.</p>
        </div>
        <div className="intro-aside">
          <span className="intro-index">01 <span>/ 03</span></span>
          <p>ASSESS<br />CONDITION</p>
          <ArrowRight size={18} />
        </div>
      </section>

      <section className="workspace-grid" aria-label="Tyre health assessment">
        <div className="assessment-panel">
          <div className="section-heading">
            <div>
              <span className="section-kicker">INPUT PARAMETERS</span>
              <h2>Vehicle &amp; tyre data</h2>
            </div>
            <span className="required-note">ALL FIELDS REQUIRED</span>
          </div>
          <form onSubmit={handleSubmit}>
            <div className="field-grid">
              {FIELDS.map((field, index) => (
                <label className="field" htmlFor={field.key} key={field.key}>
                  <span className="field-top"><span className="field-index">0{index + 1}</span>{field.label}</span>
                  <span className="input-shell">
                    <input
                      id={field.key}
                      name={field.key}
                      type="number"
                      inputMode="decimal"
                      min={field.min}
                      max={field.max}
                      step={field.step}
                      required
                      value={values[field.key]}
                      onChange={(event) => setValues((current) => ({ ...current, [field.key]: event.target.value }))}
                      aria-label={`${field.label} in ${field.unit}`}
                    />
                    <span className="input-unit">{field.unit}</span>
                  </span>
                  <span className="field-range">{field.min.toLocaleString()} – {field.max.toLocaleString()}</span>
                </label>
              ))}
            </div>
            {error && <div className="error-message" role="alert"><TriangleAlert size={16} />{error}</div>}
            <div className="form-footer">
              <span className="form-note"><CircleHelp size={15} /> Estimates are informational, not a safety inspection.</span>
              <button className="submit-button" type="submit" disabled={loading || apiOnline === false}>
                {loading ? <LoaderCircle size={17} className="spin" /> : <Activity size={17} />}
                {loading ? 'Analysing' : 'Run assessment'}
                {!loading && <ArrowRight size={16} />}
              </button>
            </div>
          </form>
        </div>

        <aside className="result-panel" aria-live="polite">
          <div className="section-heading result-heading">
            <div>
              <span className="section-kicker">MODEL OUTPUT</span>
              <h2>Health estimate</h2>
            </div>
            {result && <span className="model-tag">MODEL {result.model_version}</span>}
          </div>
          {result ? (
            <>
              <div className="result-main">
                <GaugeMeter value={result.remaining_life_percentage} status={result.health_status} />
                <div className="result-copy">
                  <span className="result-label">REMAINING LIFE</span>
                  <span className={`status-badge ${statusClass(result.health_status)}`}><span />{result.health_status}</span>
                  <span className="result-time">Updated just now</span>
                </div>
              </div>
              <div className="insights">
                <div className="insights-title"><span>SHAP INSIGHTS</span><span>TOP CONTRIBUTORS</span></div>
                <ul>
                  {result.top_factors.map((factor) => <li key={factor}><span className="insight-mark"><Check size={12} /></span>{factor}</li>)}
                </ul>
              </div>
            </>
          ) : (
            <div className="result-empty">
              <div className="empty-ring"><Gauge size={25} strokeWidth={1.4} /></div>
              <p>Your estimate will<br />appear here.</p>
              <span>Enter tyre details and run an assessment.</span>
            </div>
          )}
          <div className="result-foot"><ShieldCheck size={15} /><span>Prediction clipped to a 0–100% range</span></div>
        </aside>
      </section>

      <section className="history-section" aria-label="Assessment history">
        <div className="history-heading">
          <div>
            <span className="section-kicker">SESSION OVERVIEW</span>
            <h2>Assessment history</h2>
          </div>
          <button className="export-button" type="button" onClick={exportHistory} disabled={!history.length} title="Export assessment history as CSV">
            <ArrowDownToLine size={16} /> Export CSV
          </button>
        </div>
        <div className="history-grid">
          <div className="trend-panel">
            <div className="panel-label"><span>REMAINING LIFE TREND</span><span>{history.length} {history.length === 1 ? 'ASSESSMENT' : 'ASSESSMENTS'}</span></div>
            <TrendChart history={history} />
          </div>
          <div className="latest-panel">
            <span className="panel-label">LATEST ASSESSMENT</span>
            {latest ? (
              <div className="latest-detail">
                <strong>{latest.prediction.remaining_life_percentage}<small>%</small></strong>
                <span className={`status-badge ${statusClass(latest.prediction.health_status)}`}><span />{latest.prediction.health_status}</span>
                <span className="latest-date">{new Date(latest.createdAt).toLocaleString()}</span>
              </div>
            ) : (
              <div className="latest-empty"><RefreshCw size={17} /><span>No assessment recorded</span></div>
            )}
          </div>
        </div>
      </section>

      <footer className="footer">
        <span>TREAD/LINE <span className="footer-separator">·</span> TYRE DEGRADATION PREDICTOR</span>
        <span><span className="footer-dot" /> SYNTHETIC MODEL <span className="footer-separator">·</span> NOT FOR SAFETY-CRITICAL USE</span>
      </footer>
    </main>
  )
}

export default App
