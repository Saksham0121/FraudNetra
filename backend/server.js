const express = require("express");
const cors = require("cors");
const dotenv = require("dotenv");

dotenv.config();

const app = express();

const PORT = process.env.PORT || 5000;
const ML_API_BASE_URL = process.env.ML_API_BASE_URL || "http://127.0.0.1:8000";

app.use(cors());
app.use(express.json());

async function forwardToMlApi(path, options = {}) {
  const response = await fetch(`${ML_API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {})
    },
    ...options
  });

  const contentType = response.headers.get("content-type") || "";
  const responseBody = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const error = new Error(`ML API request failed with status ${response.status}`);
    error.status = response.status;
    error.payload = responseBody;
    throw error;
  }

  return responseBody;
}

app.get("/", async (_req, res) => {
  try {
    const mlStatus = await forwardToMlApi("/");
    res.json({
      message: "FraudNetra backend running",
      ml_api: mlStatus
    });
  } catch (error) {
    res.status(503).json({
      message: "FraudNetra backend running",
      ml_api: {
        available: false,
        error: error.payload || error.message
      }
    });
  }
});

app.post("/api/fraud/predict", async (req, res) => {
  try {
    const prediction = await forwardToMlApi("/predict", {
      method: "POST",
      body: JSON.stringify(req.body)
    });

    res.json(prediction);
  } catch (error) {
    res.status(error.status || 500).json({
      message: "Failed to get fraud prediction",
      error: error.payload || error.message
    });
  }
});

app.get("/api/fraud/stats", async (_req, res) => {
  try {
    const stats = await forwardToMlApi("/fraud-stats");
    res.json(stats);
  } catch (error) {
    res.status(error.status || 500).json({
      message: "Failed to fetch fraud stats",
      error: error.payload || error.message
    });
  }
});

app.get("/api/fraud/recent-frauds", async (req, res) => {
  const searchParams = new URLSearchParams();

  if (req.query.limit) {
    searchParams.set("limit", req.query.limit);
  }

  const querySuffix = searchParams.toString() ? `?${searchParams.toString()}` : "";

  try {
    const recentFrauds = await forwardToMlApi(`/recent-frauds${querySuffix}`);
    res.json(recentFrauds);
  } catch (error) {
    res.status(error.status || 500).json({
      message: "Failed to fetch recent fraud transactions",
      error: error.payload || error.message
    });
  }
});

app.get("/api/fraud/analytics", async (_req, res) => {
  try {
    const analytics = await forwardToMlApi("/fraud-analytics");
    res.json(analytics);
  } catch (error) {
    res.status(error.status || 500).json({
      message: "Failed to fetch fraud analytics",
      error: error.payload || error.message
    });
  }
});

if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`FraudNetra backend listening on port ${PORT}`);
  });
}

module.exports = app;
