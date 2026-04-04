import './App.css'

const metricCards = [
  {
    label: 'Transactions Today',
    value: '12,840',
    detail: 'Streaming from the backend analytics layer.'
  },
  {
    label: 'Fraud Alerts',
    value: '184',
    detail: 'High-confidence anomalies isolated for review.'
  },
  {
    label: 'Fraud Rate',
    value: '1.43%',
    detail: 'Dynamic thresholding backed by the autoencoder.'
  }
]

const feedItems = [
  {
    title: 'Recent fraud activity',
    value: 'Shopping POS spike in urban corridor',
    meta: 'Top flagged category over the latest scoring window'
  },
  {
    title: 'Model status',
    value: 'Inference pipeline healthy',
    meta: 'Artifacts loaded and FastAPI service responding'
  },
  {
    title: 'Storage',
    value: 'MongoDB transaction logging enabled',
    meta: 'Prediction history available for analytics views'
  }
]

function App() {
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
          <span className="status-pill">ML API live</span>
          <span className="status-pill">MongoDB logging on</span>
          <span className="status-pill">Backend proxy ready</span>
        </div>
      </section>

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
            <p className="panel-label">System Feed</p>
            <h3>Deployment readiness</h3>
          </div>
          <div className="feed-list">
            {feedItems.map((item) => (
              <div className="feed-item" key={item.title}>
                <p className="feed-title">{item.title}</p>
                <strong>{item.value}</strong>
                <p className="feed-meta">{item.meta}</p>
              </div>
            ))}
          </div>
        </article>
      </section>
    </main>
  )
}

export default App
