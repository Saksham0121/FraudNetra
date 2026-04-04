import { useEffect, useState } from 'react'
import './App.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000'

const initialAnalytics = {
  total_transactions: 0,
  fraud_detected: 0,
  fraud_rate_percent: 0,
  average_anomaly_score: 0
}

function formatNumber(value) {
  return new Intl.NumberFormat('en-IN').format(value)
}

function formatPercent(value) {
  return `${Number(value).toFixed(2)}%`
}

function formatScore(value) {
  return Number(value).toFixed(2)
}

function formatTimestamp(value) {
  if (!value) {
    return 'Not available'
  }

  return new Date(value).toLocaleString('en-IN', {
    dateStyle: 'medium',
    timeStyle: 'short'
  })
}

function App() {
  const [analytics, setAnalytics] = useState(initialAnalytics)
  const [recentFrauds, setRecentFrauds] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true

    async function loadDashboardData() {
      try {
        setLoading(true)
        setError('')

        const [analyticsResponse, recentFraudsResponse] = await Promise.all([
          fetch(`${API_BASE_URL}/api/fraud/analytics`),
          fetch(`${API_BASE_URL}/api/fraud/recent-frauds?limit=6`)
        ])

        if (!analyticsResponse.ok || !recentFraudsResponse.ok) {
          throw new Error('Backend routes did not return a successful response.')
        }

        const analyticsPayload = await analyticsResponse.json()
        const recentFraudsPayload = await recentFraudsResponse.json()

        if (!active) {
          return
        }

        setAnalytics(analyticsPayload)
        setRecentFrauds(recentFraudsPayload.recent_frauds || [])
      } catch (fetchError) {
        if (!active) {
          return
        }

        setError(fetchError.message || 'Failed to load dashboard data.')
      } finally {
        if (active) {
          setLoading(false)
        }
      }
    }

    loadDashboardData()

    const intervalId = window.setInterval(loadDashboardData, 10000)

    return () => {
      active = false
      window.clearInterval(intervalId)
    }
  }, [])

  const metricCards = [
    {
      label: 'Transactions',
      value: formatNumber(analytics.total_transactions),
      detail: 'Processed transaction records available to the dashboard.'
    },
    {
      label: 'Fraud Alerts',
      value: formatNumber(analytics.fraud_detected),
      detail: 'Predictions classified as anomalous by the ML service.'
    },
    {
      label: 'Fraud Rate',
      value: formatPercent(analytics.fraud_rate_percent),
      detail: 'Share of logged transactions currently flagged as fraud.'
    },
    {
      label: 'Avg Anomaly Score',
      value: formatScore(analytics.average_anomaly_score),
      detail: 'Mean reconstruction-error score across stored predictions.'
    }
  ]

  return (
    <main className="dashboard-shell">
      <section className="hero-panel">
        <div className="hero-copy">
          <p className="eyebrow">FraudNetra Dashboard</p>
          <h1>Real-time fraud intelligence for transaction monitoring.</h1>
          <p className="hero-text">
            A frontend shell for the fraud detection platform, designed to plug
            into the backend analytics routes and visualize anomaly trends.
          </p>
        </div>
        <div className="hero-status">
          <span className="status-pill">
            {loading ? 'Refreshing dashboard' : 'Backend data live'}
          </span>
          <span className="status-pill">MongoDB analytics ready</span>
          <span className="status-pill">Refresh interval: 10s</span>
        </div>
      </section>

      {error ? <p className="banner banner-error">{error}</p> : null}
      {!error && loading ? <p className="banner">Loading fraud metrics...</p> : null}

      <section className="metrics-grid">
        {metricCards.map((card) => (
          <article className="metric-card" key={card.label}>
            <p className="metric-label">{card.label}</p>
            <h2>{card.value}</h2>
            <p className="metric-detail">{card.detail}</p>
          </article>
        ))}
      </section>

      <section className="insight-grid">
        <article className="panel panel-large">
          <div className="panel-heading">
            <p className="panel-label">Pipeline Overview</p>
            <h3>Fraud detection workflow</h3>
          </div>
          <div className="workflow">
            <span>Transaction Input</span>
            <span>Preprocessing</span>
            <span>Autoencoder</span>
            <span>Threshold Check</span>
            <span>Fraud Decision</span>
          </div>
        </article>

        <article className="panel">
          <div className="panel-heading">
            <p className="panel-label">Recent Alerts</p>
            <h3>Suspicious transaction feed</h3>
          </div>
          {recentFrauds.length === 0 ? (
            <p className="empty-state">
              No fraud records available yet. Send predictions through the API to
              populate this monitoring panel.
            </p>
          ) : (
            <div className="feed-list">
              {recentFrauds.map((item, index) => (
                <div className="feed-item" key={`${item.timestamp}-${index}`}>
                  <div className="feed-row">
                    <p className="feed-title">{item.transaction.category}</p>
                    <span className="feed-badge">Fraud</span>
                  </div>
                  <strong>
                    Amount: {formatNumber(item.transaction.amt)} | Score:{' '}
                    {formatScore(item.anomaly_score)}
                  </strong>
                  <p className="feed-meta">{formatTimestamp(item.timestamp)}</p>
                </div>
              ))}
            </div>
          )}
        </article>
      </section>
    </main>
  )
}

export default App
