"""
Blood Demand Prediction AI Service
=====================================
مصدر البيانات الوحيد: جدول BloodBags

  Status = 0 (Available)  → المخزون الحالي
  Status = 1 (Withdrawn)  → تاريخ الاستهلاك (لتدريب الموديل)

BloodType enum:
  1=A+  2=A-  3=B+  4=B-  5=AB+  6=AB-  7=O+  8=O-

الموديل: RandomForest Regressor
الدقة:   MAE / RMSE / MAPE (train-test split زمني حقيقي)

تشغيل:
  pip install -r requirements.txt
  uvicorn main:app --reload --port 8001
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union
import math
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

app = FastAPI(title="Blood Demand Prediction AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Constants ────────────────────────────────────────────────────────────────

LAGS         = [1, 2, 3, 7, 14]
ROLL_WINDOWS = [7, 14]
MIN_POINTS   = 40
RANDOM_STATE = 42

AVAILABLE_STATUS = 0   # BloodBagStatus.Available  → مخزون
USED_STATUS      = 1   # BloodBagStatus.Withdrawn  → استهلاك

# BloodType enum بالظبط زي ما عندك في الـ .NET
BLOOD_TYPE_LABELS = {
    1: "A+",  2: "A-",
    3: "B+",  4: "B-",
    5: "AB+", 6: "AB-",
    7: "O+",  8: "O-",
}

def label_blood_type(value) -> str:
    """بتحوّل BloodType int → اسم مقروء. لو string بترجعها زي ما هي."""
    try:
        return BLOOD_TYPE_LABELS.get(int(value), str(value))
    except (ValueError, TypeError):
        return str(value)

# ─── Models ───────────────────────────────────────────────────────────────────

class BloodBagItem(BaseModel):
    blood_type:    Union[int, str]       # int enum من الداتابيز
    status:        int                   # 0=Available / 1=Withdrawn
    created_at:    str                   # ISO datetime — تاريخ إنشاء الكيس
    withdrawn_at:  Optional[str] = None  # ISO datetime — تاريخ السحب الفعلي
    expiry_date:   Optional[str] = None  # ISO datetime أو None

class PredictRequest(BaseModel):
    hospital_id:  int = Field(..., gt=0,
                              description="معرّف المستشفى — لازم يكون موجود وأكبر من 0")
    blood_bags:   List[BloodBagItem]   # كل الأكياس (Status 0 و 1)
    horizon_days: int  = Field(default=7, ge=1, le=180,
                               description="الفترة بالأيام (1–180)")

    @validator("blood_bags")
    def bags_not_empty(cls, v):
        if not v:
            raise ValueError("blood_bags لازم تكون فيه أكياس — مفيش بيانات للمستشفى ده")
        return v

class BloodTypePrediction(BaseModel):
    blood_type:        str
    method:            str              # ml_random_forest | statistical_fallback
    predicted_total:   float            # إجمالي الاستهلاك المتوقّع
    predicted_per_day: List[float]      # توزيع يومي
    accuracy_percent:  Optional[float]  # None لو الداتا قليلة
    mae:               Optional[float]
    rmse:              Optional[float]
    mape:              Optional[float]
    current_stock:     float            # أكياس Available الصالحة
    units_required:    float            # وحدات محتاج تطلبها
    units_surplus:     float            # وحدات زيادة
    days_of_coverage:  Optional[float]  # المخزون يكفي كام يوم
    shortage_expected: bool

class PredictResponse(BaseModel):
    hospital_id:              int
    horizon_days:             int
    demand_level:             str               # Low | Medium | High
    total_expected_units:     float
    total_units_required:     float
    overall_accuracy_percent: Optional[float]
    warnings:                 List[str]
    predictions:              List[BloodTypePrediction]

# ─── Current Stock ────────────────────────────────────────────────────────────

def calc_current_stock(blood_bags: List[BloodBagItem]) -> dict:
    """
    بيحسب المخزون الحالي من الأكياس المتاحة (Status = 0).
    الأكياس المنتهية صلاحيتها بتتستبعد.
    كل كيس = وحدة واحدة.
    """
    stock = {}
    now   = datetime.utcnow()

    for bag in blood_bags:
        if bag.status != AVAILABLE_STATUS:
            continue
        if bag.expiry_date:
            try:
                exp = datetime.fromisoformat(bag.expiry_date.replace("Z", ""))
                if exp <= now:
                    continue
            except ValueError:
                pass
        bt = label_blood_type(bag.blood_type)
        stock[bt] = stock.get(bt, 0) + 1

    return stock

# ─── Consumption History ──────────────────────────────────────────────────────

def build_daily_consumption(blood_bags: List[BloodBagItem], blood_type: str):
    """
    بيبني سلسلة استهلاك يومية من الأكياس المسحوبة (Status = 1).
    بيستخدم withdrawn_at (تاريخ السحب الفعلي)، أو created_at كـ fallback.
    كل كيس مسحوب = وحدة استهلاك واحدة.
    الأيام اللي مفيهاش سحب بتتملي بصفر.
    """
    daily = {}

    for bag in blood_bags:
        if label_blood_type(bag.blood_type) != blood_type:
            continue
        if bag.status != USED_STATUS:
            continue
        # بنستخدم withdrawn_at (تاريخ السحب الفعلي) لو موجود، وإلا created_at
        date_str = bag.withdrawn_at or bag.created_at
        try:
            day = datetime.fromisoformat(date_str.replace("Z", "")).date()
        except (ValueError, AttributeError):
            continue
        daily[day] = daily.get(day, 0) + 1

    if not daily:
        return [], []

    start, end = min(daily), max(daily)
    current, dates, values = start, [], []
    while current <= end:
        dates.append(current)
        values.append(float(daily.get(current, 0)))
        current += timedelta(days=1)

    return dates, values

# ─── Feature Engineering ──────────────────────────────────────────────────────

def _feature_row(values: list, dates: list, i: int) -> dict:
    """
    Features:
      lag_1, 2, 3, 7, 14   → استهلاك أيام فاتوا
      roll_mean_7/14        → متوسط متحرك
      roll_std_7/14         → تذبذب الاستهلاك
      dow_0..6              → يوم الأسبوع (one-hot)
      is_weekend            → هل عطلة
      week_of_month         → أنهي أسبوع في الشهر
    """
    row = {}

    for lag in LAGS:
        row[f"lag_{lag}"] = values[i - lag]

    for w in ROLL_WINDOWS:
        window = values[i - w:i]
        row[f"roll_mean_{w}"] = float(np.mean(window)) if window else 0.0
        row[f"roll_std_{w}"]  = float(np.std(window))  if len(window) > 1 else 0.0

    dow = dates[i].weekday()
    for d in range(7):
        row[f"dow_{d}"] = 1.0 if d == dow else 0.0

    row["is_weekend"]    = 1.0 if dow >= 5 else 0.0
    row["week_of_month"] = float((dates[i].day - 1) // 7 + 1)

    return row


def _build_xy(values: list, dates: list):
    start_idx = max(max(LAGS), max(ROLL_WINDOWS))
    rows, targets = [], []
    for i in range(start_idx, len(values)):
        rows.append(_feature_row(values, dates, i))
        targets.append(values[i])
    return pd.DataFrame(rows), np.array(targets, dtype=float)

# ─── Accuracy Metrics ─────────────────────────────────────────────────────────

def _calc_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(math.sqrt(np.mean((y_true - y_pred) ** 2)))

    mask = y_true > 0
    if mask.sum() > 0:
        mape     = float(np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100)
        accuracy = max(0.0, 100.0 - mape)
    else:
        mape, accuracy = None, None

    return {
        "mae":              round(mae, 3),
        "rmse":             round(rmse, 3),
        "mape":             round(mape, 2)     if mape     is not None else None,
        "accuracy_percent": round(accuracy, 2) if accuracy is not None else None,
    }

# ─── Forecaster ───────────────────────────────────────────────────────────────

@dataclass
class BloodTypeForecaster:
    blood_type:  str
    model:       object = None
    method:      str    = "ml_random_forest"
    metrics:     dict   = field(default_factory=dict)
    last_values: list   = field(default_factory=list)
    last_dates:  list   = field(default_factory=list)

    def fit(self, dates: list, values: list) -> "BloodTypeForecaster":
        self.last_dates, self.last_values = dates, values

        if len(values) < MIN_POINTS:
            self.method  = "statistical_fallback"
            self.model   = None
            self.metrics = {"mae":None,"rmse":None,"mape":None,"accuracy_percent":None}
            return self

        X, y = _build_xy(values, dates)

        test_size = max(5, int(0.2 * len(X)))
        X_tr, X_te = X.iloc[:-test_size], X.iloc[-test_size:]
        y_tr, y_te = y[:-test_size],      y[-test_size:]

        eval_m = RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2,
            max_features="sqrt", random_state=RANDOM_STATE
        )
        eval_m.fit(X_tr, y_tr)
        self.metrics = _calc_metrics(y_te, eval_m.predict(X_te))

        self.model = RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2,
            max_features="sqrt", random_state=RANDOM_STATE
        )
        self.model.fit(X, y)
        self.method = "ml_random_forest"
        return self

    def forecast(self, horizon_days: int) -> list:
        return self._forecast_ml(horizon_days) if self.model else \
               self._forecast_statistical(horizon_days)

    def _forecast_ml(self, horizon_days: int) -> list:
        values = list(self.last_values)
        dates  = list(self.last_dates)
        preds  = []
        for _ in range(horizon_days):
            next_day = dates[-1] + timedelta(days=1)
            dates.append(next_day); values.append(0.0)
            i    = len(values) - 1
            pred = max(0.0, float(self.model.predict(
                pd.DataFrame([_feature_row(values, dates, i)]))[0]))
            values[-1] = pred
            preds.append(pred)
        return preds

    def _forecast_statistical(self, horizon_days: int) -> list:
        vals  = self.last_values[-14:] if self.last_values else [0.0]
        base  = float(np.mean(vals)) if vals else 0.0
        if len(vals) >= 4:
            half  = len(vals) // 2
            trend = (np.mean(vals[half:]) - np.mean(vals[:half])) / max(1, half)
            trend = max(min(trend, base * 0.1), -(base * 0.1))
        else:
            trend = 0.0
        return [max(0.0, base + trend * (i + 1)) for i in range(horizon_days)]

# ─── Demand Level ─────────────────────────────────────────────────────────────

def calc_demand_level(total: float, horizon: int) -> str:
    avg = total / max(1, horizon)
    return "High" if avg > 100 else "Medium" if avg >= 50 else "Low"

# ─── Main Endpoint ────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    يستقبل من الـ .NET:
      - blood_bags  → كل أكياس الدم بتاعة المستشفى
          Status=0  → Available  → المخزون الحالي
          Status=1  → Withdrawn  → تاريخ الاستهلاك للتدريب
      - horizon_days → الفترة (افتراضي 7 أيام)
    """
    horizon  = request.horizon_days or 7

    # ── التحقق إن المستشفى عنده بيانات ──────────────────────────────────────
    if not request.blood_bags:
        raise HTTPException(
            status_code=404,
            detail=f"المستشفى رقم {request.hospital_id} مفيش له بيانات أكياس دم"
        )

    used_bags = [b for b in request.blood_bags if b.status == USED_STATUS]
    if not used_bags:
        raise HTTPException(
            status_code=422,
            detail=(
                f"المستشفى رقم {request.hospital_id} مفيش له سجلات سحب (status=1). "
                f"الموديل محتاج تاريخ استهلاك عشان يتنبأ."
            )
        )

    stock    = calc_current_stock(request.blood_bags)
    warnings = []

    # الفصايل اللي عندها سجلات سحب (Status=1)
    blood_types = list({
        label_blood_type(b.blood_type)
        for b in request.blood_bags
        if b.status == USED_STATUS
    })

    predictions = []
    for bt in blood_types:
        dates, values = build_daily_consumption(request.blood_bags, bt)
        if not values:
            continue

        fc      = BloodTypeForecaster(blood_type=bt).fit(dates, values)
        per_day = fc.forecast(horizon)
        total   = round(sum(per_day), 2)

        cur_stock      = float(stock.get(bt, 0))
        avg_daily      = total / horizon
        days_cov       = round(cur_stock / avg_daily, 1) if avg_daily > 0 else None
        shortage       = cur_stock < total
        units_required = round(max(0.0, total - cur_stock), 2)
        units_surplus  = round(max(0.0, cur_stock - total), 2)

        if fc.method == "statistical_fallback":
            warnings.append(
                f"فصيلة {bt}: أقل من {MIN_POINTS} يوم سحب — "
                f"طريقة إحصائية، الدقة غير متاحة."
            )

        predictions.append(BloodTypePrediction(
            blood_type=bt,
            method=fc.method,
            predicted_total=total,
            predicted_per_day=[round(x, 2) for x in per_day],
            accuracy_percent=fc.metrics.get("accuracy_percent"),
            mae=fc.metrics.get("mae"),
            rmse=fc.metrics.get("rmse"),
            mape=fc.metrics.get("mape"),
            current_stock=cur_stock,
            units_required=units_required,
            units_surplus=units_surplus,
            days_of_coverage=days_cov,
            shortage_expected=shortage,
        ))

    predictions.sort(key=lambda x: x.units_required, reverse=True)

    total_units = sum(p.predicted_total for p in predictions)
    accs        = [p.accuracy_percent for p in predictions
                   if p.accuracy_percent is not None]
    overall_acc = round(sum(accs)/len(accs), 2) if accs else None

    return PredictResponse(
        hospital_id=request.hospital_id,
        horizon_days=horizon,
        demand_level=calc_demand_level(total_units, horizon),
        total_expected_units=round(total_units, 2),
        total_units_required=round(sum(p.units_required for p in predictions), 2),
        overall_accuracy_percent=overall_acc,
        warnings=warnings,
        predictions=predictions,
    )


@app.get("/health")
def health():
    return {"status": "ok", "service": "Blood Demand Prediction AI"}
