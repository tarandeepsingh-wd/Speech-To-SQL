from fastapi import FastAPI
from pydantic import BaseModel

from module1_asr.sql_ops import extract_sql_operators


app = FastAPI(title="TADKA Module 1 SQL Operator Extractor")


class SqlOperatorRequest(BaseModel):
    query: str


@app.post("/extract_sql_operators")
def extract_sql_operator_endpoint(payload: SqlOperatorRequest) -> dict:
    return {
        "operators": extract_sql_operators(payload.query),
    }
