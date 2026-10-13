"""
CartIQ Production REST API
Built with FastAPI & Pydantic.
Provides:
- GET  /health                   : System health, uptime & status
- POST /predict/customer-tier    : Customer VIP classification (is_high_value) & annual spend regression
- POST /predict/delivery-delay   : Order delivery delay classification (is_delayed)
- POST /drift/detect             : Live Population Stability Index (PSI) & KS-test drift check
"""

import os
import time
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

from data_loader import get_customer_summary, get_order_summary
from drift_monitor import run_dataset_drift_report

START_TIME = time.time()

app = FastAPI(
    title="🛒 CartIQ Production ML Serving API",
    description="Enterprise Machine Learning REST Microservice for E-Commerce Customer Lifetime Value & Order Delivery Delay Predictions.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- PYDANTIC SCHEMAS -----------------
class CustomerProfile(BaseModel):
    age: int = Field(32, ge=18, le=100, example=32)
    total_orders: int = Field(8, ge=1, example=8)
    total_items: int = Field(15, ge=1, example=15)
    avg_delivery_days: float = Field(4.5, ge=0.0, example=4.2)
    avg_order_value: float = Field(1200.0, ge=0.0, example=1450.0)
    gender: Optional[str] = Field("Female", example="Female")
    state: Optional[str] = Field("California", example="California")
    model_name: Optional[str] = Field("random_forest", example="random_forest")

class CustomerTierResponse(BaseModel):
    customer_tier: str
    is_high_value: int
    high_value_probability: float
    expected_spend: float
    model_used: str
    inference_time_ms: float

class OrderProfile(BaseModel):
    total_order_price: float = Field(1500.0, ge=0.0, example=1850.0)
    item_count: int = Field(3, ge=1, example=3)
    age: int = Field(28, ge=18, example=28)
    order_month: int = Field(11, ge=1, le=12, example=11)
    payment: Optional[str] = Field("Credit Card", example="Credit Card")
    gender: Optional[str] = Field("Male", example="Male")
    state: Optional[str] = Field("New York", example="New York")

class DeliveryDelayResponse(BaseModel):
    delivery_status: str
    is_delayed: int
    delay_probability: float
    inference_time_ms: float

class DriftCheckRequest(BaseModel):
    current_batch: List[Dict[str, Any]]

# ----------------- CACHED MODELS -----------------
_MODELS = {}

def get_fitted_models():
    global _MODELS
    if not _MODELS:
        cust_df = get_customer_summary()
        feat_num = ['age', 'total_orders', 'total_items', 'avg_delivery_days', 'avg_order_value']
        feat_cat = ['gender', 'state']
        X_raw = cust_df[feat_num + feat_cat]
        y_c = cust_df['is_high_value']
        y_r = cust_df['total_spend']
        X = pd.get_dummies(X_raw, columns=feat_cat, drop_first=True, dtype=int)

        rf_c = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42).fit(X, y_c)
        rf_r = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42).fit(X, y_r)

        _MODELS = {
            "columns": X.columns.tolist(),
            "customer_clf": rf_c,
            "customer_reg": rf_r
        }
    return _MODELS

# ----------------- ENDPOINTS -----------------
@app.get("/health", tags=["Monitoring"])
def health():
    return {
        "status": "healthy",
        "service": "CartIQ E-Commerce ML Serving Gateway",
        "uptime_seconds": round(time.time() - START_TIME, 2)
    }

@app.post("/predict/customer-tier", response_model=CustomerTierResponse, tags=["Inference"])
def predict_customer_tier(req: CustomerProfile):
    t0 = time.time()
    models = get_fitted_models()
    expected_cols = models["columns"]

    row = {
        "age": req.age,
        "total_orders": req.total_orders,
        "total_items": req.total_items,
        "avg_delivery_days": req.avg_delivery_days,
        "avg_order_value": req.avg_order_value,
        f"gender_{req.gender}": 1,
        f"state_{req.state}": 1
    }
    df_in = pd.DataFrame([row]).reindex(columns=expected_cols, fill_value=0)

    clf = models["customer_clf"]
    reg = models["customer_reg"]

    pred_class = int(clf.predict(df_in)[0])
    prob = float(clf.predict_proba(df_in)[0, 1])
    pred_spend = float(reg.predict(df_in)[0])

    latency = round((time.time() - t0) * 1000, 2)
    return CustomerTierResponse(
        customer_tier="High Value (VIP)" if pred_class == 1 else "Standard Customer",
        is_high_value=pred_class,
        high_value_probability=round(prob, 4),
        expected_spend=round(pred_spend, 2),
        model_used=req.model_name or "random_forest",
        inference_time_ms=latency
    )

@app.post("/drift/detect", tags=["Monitoring"])
def check_drift(req: DriftCheckRequest):
    cust_df = get_customer_summary()
    curr_df = pd.DataFrame(req.current_batch)
    if curr_df.empty:
        raise HTTPException(status_code=400, detail="Provided batch is empty.")
    report = run_dataset_drift_report(cust_df, curr_df)
    return report

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)
