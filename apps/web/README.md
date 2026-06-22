# Speech-to-SQL Web Console

This React workbench is the UI demo surface for the Speech-to-SQL pipeline. The
default browser path uses the local Node/Express proxy, which forwards requests
to the Python ML/SQL API.

Start the Python API:

```powershell
uvicorn apps.api.nlu_api:app --reload --port 8000
```

Start the Node proxy:

```powershell
npm --prefix apps\proxy install
npm --prefix apps\proxy start
```

Start the web app:

```powershell
npm --prefix apps\web install
npm --prefix apps\web start
```

By default the app calls `http://localhost:5000/api/query`. Override that with:

```powershell
$env:REACT_APP_API_BASE="http://localhost:8000"
```
