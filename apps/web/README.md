# Speech-to-SQL Web Console

This is a small React console for the local Speech-to-SQL API. It is not the
main demo path; the CLI remains the most complete module-by-module view.

Start the Python API:

```powershell
uvicorn apps.api.nlu_api:app --reload --port 8000
```

Start the web app:

```powershell
npm --prefix apps\web install
npm --prefix apps\web start
```

By default the app calls `http://localhost:8000/query`. Override that with:

```powershell
$env:REACT_APP_API_BASE="http://localhost:8000"
```
