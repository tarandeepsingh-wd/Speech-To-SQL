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

// Proxy route: forwards app API calls to the Python ML/SQL service.
app.use('/api', async (req, res) => {
  try {
    const axios = require('axios');
    const response = await axios({
      method: req.method,
      url: `${PYTHON_API}${req.originalUrl.replace(/^\/api/, '')}`,
      data: req.body,
      params: req.query,
      validateStatus: () => true,
    });
    res.status(response.status);
    res.json(response.data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Node proxy running on http://localhost:${PORT}`);
});
