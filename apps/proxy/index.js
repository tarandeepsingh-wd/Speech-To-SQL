const express = require('express');
const cors = require('cors');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());

const PYTHON_API = process.env.PYTHON_API || 'http://localhost:8000';

// Health check
app.get('/', (req, res) => {
  res.json({ status: 'Speech-to-SQL Node proxy running' });
});

// Proxy route: forwards queries to the Python API.
app.post('/api/query', async (req, res) => {
  try {
    const axios = require('axios');
    const response = await axios.post(
      `${PYTHON_API}/query`,
      req.body
    );
    res.json(response.data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Node proxy running on http://localhost:${PORT}`);
});
