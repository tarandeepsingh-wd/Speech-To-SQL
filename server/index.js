const express = require('express');
const cors = require('cors');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());

// Health check
app.get('/', (req, res) => {
  res.json({ status: 'TADKA Node server running 🌶️' });
});

// Proxy route → forwards queries to Python pipeline
app.post('/api/query', async (req, res) => {
  try {
    const axios = require('axios');
    const response = await axios.post(
      `${process.env.PYTHON_API}/query`,
      req.body
    );
    res.json(response.data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`✅ Node server running on http://localhost:${PORT}`);
});