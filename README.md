# FraudNetra

FraudNetra is a multi-part fraud detection project featuring an ML microservice, a Node.js backend proxy layer, and a React dashboard frontend.

## Architecture

1. **ML Service (Python + FastAPI)**: Handles data preprocessing, feature engineering, and inference using a trained Autoencoder model to calculate anomaly scores and detect fraud. It also maintains in-memory statistics and connects to MongoDB for storing transaction logs.
2. **Backend Proxy (Node.js + Express)**: Serves as a gateway between the frontend and the ML service, abstracting away the ML endpoints and applying necessary configurations.
3. **Frontend Dashboard (React + Vite)**: A dynamic user interface that fetches live metrics, recent frauds, and pipeline analytics to visually monitor the fraud detection system.
4. **Database (MongoDB)**: Used to persist transaction logs including their anomaly score and fraud prediction results.

## Prerequisites

- Node.js v20+
- Python 3.11+
- MongoDB 8.0+
- Docker & Docker Compose (Optional, for containerized multi-service deployment)

## Local Setup

### 1. Start MongoDB
Run a local MongoDB instance. If you installed via Homebrew:
```bash
brew services start mongodb-community@8.0
```

### 2. Start ML Microservice
```bash
cd ml-api
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```
Runs at: `http://127.0.0.1:8000`

### 3. Start Backend Proxy
```bash
cd backend
npm install
npm start
```
Runs at: `http://127.0.0.1:5000`

### 4. Start Frontend
```bash
cd frontend
npm install
npm run dev
```
Runs at: `http://127.0.0.1:5173` (Vite Default)

## Docker Compose Setup

Run the entire application stack using Docker.

```bash
docker-compose up --build
```

- Frontend App: `http://localhost:80`
- Backend API: `http://localhost:5001`
- ML API Docs: `http://localhost:8000/docs`
- MongoDB: `localhost:27017`

## API Routes

### ML Service (`:8000`)
- `GET /`: Health check.
- `POST /predict`: Submit a transaction JSON payload to get a fraud prediction.
- `GET /fraud-stats`: Returns global processing and fraud count statistics.
- `GET /recent-frauds`: Returns recent transactions flagged as fraud.
- `GET /fraud-analytics`: Returns aggregated analytics such as fraud rate and categorizations.

### Backend Proxy (`:5000`)
- `GET /`: Backend root / Health check.
- `POST /api/fraud/predict`: Proxies `POST /predict`.
- `GET /api/fraud/stats`: Proxies `GET /fraud-stats`.
- `GET /api/fraud/recent-frauds`: Proxies `GET /recent-frauds?limit=x`.
- `GET /api/fraud/analytics`: Proxies `GET /fraud-analytics`.

## Environment Variables

- **`ml-api/.env`** (optional): `MONGODB_URI`, `MONGODB_DB_NAME`
- **`backend/.env`**: `PORT=5000`, `ML_API_BASE_URL=http://127.0.0.1:8000`
- **`frontend/.env`**: `VITE_API_BASE_URL=http://127.0.0.1:5000`
