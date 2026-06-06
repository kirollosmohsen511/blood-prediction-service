"""
Blood Demand Prediction AI Service
=====================================
بيتنبأ باستهلاك الدم لكل فصيلة بناءً على التاريخ.

العوامل:
  - lag features   → استهلاك أيام فاتوا (1, 2, 3, 7)
  - rolling avg    → متوسط آخر 7 أيام
  - day of week    → أنهي يوم في الأسبوع (seasonal pattern)
  الموديل: RandomForest Regressor
  الدقة: MAE / RMSE / MAPE (train-test split زمني حقيقي)

تشغيل:
  pip install -r requirements.txt
  uvicorn main:app --reload --port 8001
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
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

LAGS        = [1, 2, 3, 7]   # أيام اللي بنرجع ليها كـ features
ROLL_WINDOW = 7               # نافذة المتوسط المتحرك
MIN_POINTS  = 35              # أقل عدد أيام عشان ندرّب موديل ML بأمانة
RANDOM_STATE = 42

# ─── Models ───────────────────────────────────────────────────────────────────

class InventoryLogItem(BaseModel):
    blood_type: str      # "A+", "O-", ...
    change_amount: int   # سالب = استهلاك (دم خرج) / موجب = توريد (دم دخل)
    changed_at: str      # ISO datetime string  "2026-01-15T10:00:00"

class BloodBagItem(BaseModel):
    blood_type: str
    expiry_date: Optional[str] = None   # ISO datetime string أو None

class PredictRequest(BaseModel):
    hospital_id: int
    inventory_logs: List[InventoryLogItem]  # من جدول InventoryLogs
    blood_bags: List[BloodBagItem]          # من جدول BloodBags (Status=Available فقط)
    horizon_days: Optional[int] = 7        # الفترة المطلوب التنبؤ بها

class BloodTypePrediction(BaseModel):
    blood_type: str
    method: str                          # ml_random_forest | statistical_fallback
    predicted_total: float               # إجمالي الاستهلاك المتوقّع خلال الفترة
    predicted_per_day: List[float]       # توزيع يومي
    accuracy_percent: Optional[float]    # None لو الداتا قليلة (مش بنخترع رقم)
    mae: Optional[float]
    rmse: Optional[float]
    mape: Optional[float]
    current_stock: float                 # عدد الأكياس المتاحة الصالحة
    days_of_coverage: Optional[float]    # المخزون يكفي كام يوم
    shortage_expected: bool              # هل المخزون أقل من المتوقّع؟

class PredictResponse(BaseModel):
    hospital_id: int
    horizon_days: int
    demand_level: str                        # Low | Medium | High
    total_expected_units: float              # مجموع كل الفصايل
    overall_accuracy_percent: Optional[float]
    warnings: List[str]
    predictions: List[BloodTypePrediction]   # مرتبة حسب الاستهلاك تنازلياً

# ─── Current Stock from BloodBags ─────────────────────────────────────────────

def calc_current_stock(blood_bags: List[BloodBagItem]) -> dict:
    """
    بيحسب المخزون الحالي من قائمة الأكياس المتاحة.
    كل كيس = وحدة واحدة.
    الأكياس المنتهية صلاحيتها بتتستبعد.
    """
    stock = {}
    now = datetime.utcnow()

    for bag in blood_bags:
        # استبعاد المنتهية الصلاحية
        if bag.expiry_date:
            try:
                exp = datetime.fromisoformat(bag.expiry_date.replace("Z", ""))
                if exp <= now:
                    continue
            except ValueError:
                pass

        bt = bag.blood_type.strip()
        stock[bt] = stock.get(bt, 0) + 1

    return stock

# ─── Series Builder ───────────────────────────────────────────────────────────

def build_daily_consumption(logs: List[InventoryLogItem], blood_type: str):
    """
    بيبني سلسلة استهلاك يومية لفصيلة معيّنة.
    الاستهلاك = مجموع القيم السالبة في change_amount (الدم اللي خرج).
    الأيام اللي مفيهاش حركة بتتملي بصفر عشان السلسلة تفضل متصلة.
    """
    daily = {}
    for log in logs:
        if log.blood_type.strip() != blood_type:
            continue
        if log.change_amount >= 0:
            continue  # توريد مش استهلاك — بنشيله

        try:
            day = datetime.fromisoformat(log.changed_at.replace("Z", "")).date()
        except ValueError:
            continue

        daily[day] = daily.get(day, 0) + (-log.change_amount)

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
    بيبني صف features للنقطة رقم i:
      - lag_1, lag_2, lag_3, lag_7  →  استهلاك أيام فاتوا
      - roll_mean_7                  →  متوسط آخر 7 أيام
      - dow_0 .. dow_6               →  يوم الأسبوع (one-hot)
    """
    row = {}
    for lag in LAGS:
        row[f"lag_{lag}"] = values[i - lag]
    row["roll_mean_7"] = float(sum(values[i - ROLL_WINDOW:i]) / ROLL_WINDOW)
    dow = dates[i].weekday()
    for d in range(7):
        row[f"dow_{d}"] = 1.0 if d == dow else 0.0
    return row


def _build_xy(values: list, dates: list):
    """بيبني الـ feature matrix X والـ target vector y للتدريب."""
    start_idx = max(max(LAGS), ROLL_WINDOW)
    rows, targets = [], []
    for i in range(start_idx, len(values)):
        rows.append(_feature_row(values, dates, i))
        targets.append(values[i])
    return pd.DataFrame(rows), np.array(targets, dtype=float)

# ─── Accuracy Metrics ─────────────────────────────────────────────────────────

def _calc_metrics(y_true, y_pred) -> dict:
    """
    بيحسب MAE / RMSE / MAPE / accuracy% على الـ test set.
    MAPE بيتحسب على الأيام اللي الاستهلاك فيها > 0 بس (عشان نتجنب قسمة على صفر).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(math.sqrt(np.mean((y_true - y_pred) ** 2)))

    mask = y_true > 0
    if mask.sum() > 0:
        mape     = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        accuracy = max(0.0, 100.0 - mape)
    else:
        mape, accuracy = None, None

    return {
        "mae":              round(mae, 3),
        "rmse":             round(rmse, 3),
        "mape":             round(mape, 2)     if mape     is not None else None,
        "accuracy_percent": round(accuracy, 2) if accuracy is not None else None,
    }

# ─── Blood Type Forecaster ────────────────────────────────────────────────────

@dataclass
class BloodTypeForecaster:
    blood_type: str
    model:       object = None
    method:      str    = "ml_random_forest"
    metrics:     dict   = field(default_factory=dict)
    last_values: list   = field(default_factory=list)
    last_dates:  list   = field(default_factory=list)

    def fit(self, dates: list, values: list) -> "BloodTypeForecaster":
        """
        بيدرّب الموديل ويحسب الدقة.
        لو الداتا أقل من MIN_POINTS → ينزل على طريقة إحصائية
        من غير ما يخترع رقم دقة.
        """
        self.last_dates, self.last_values = dates, values

        if len(values) < MIN_POINTS:
            # ── داتا قليلة: statistical fallback ──────────────────────
            self.method  = "statistical_fallback"
            self.model   = None
            self.metrics = {"mae": None, "rmse": None,
                            "mape": None, "accuracy_percent": None}
            return self

        # ── داتا كافية: تدريب RandomForest ────────────────────────────
        X, y = _build_xy(values, dates)

        # تقسيم زمني: آخر 20% للاختبار (بدون خلط — ده time series)
        test_size = max(5, int(0.2 * len(X)))
        X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
        y_train, y_test = y[:-test_size],      y[-test_size:]

        # 1) موديل للتقييم (على الـ test set)
        eval_model = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, random_state=RANDOM_STATE
        )
        eval_model.fit(X_train, y_train)
        self.metrics = _calc_metrics(y_test, eval_model.predict(X_test))

        # 2) موديل نهائي (على كل الداتا للتنبؤ)
        self.model = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, random_state=RANDOM_STATE
        )
        self.model.fit(X, y)
        self.method = "ml_random_forest"
        return self

    def forecast(self, horizon_days: int) -> list:
        if self.model is not None:
            return self._forecast_ml(horizon_days)
        return self._forecast_statistical(horizon_days)

    def _forecast_ml(self, horizon_days: int) -> list:
        """
        تنبؤ متسلسل: يتوقّع يوم، يضيفه للتاريخ، يستخدمه لتوقّع التالي.
        """
        values = list(self.last_values)
        dates  = list(self.last_dates)
        preds  = []

        for _ in range(horizon_days):
            next_day = dates[-1] + timedelta(days=1)
            dates.append(next_day)
            values.append(0.0)       # placeholder لبناء الـ feature row
            i    = len(values) - 1
            row  = _feature_row(values, dates, i)
            pred = max(0.0, float(self.model.predict(pd.DataFrame([row]))[0]))
            values[-1] = pred        # بنستبدل الـ placeholder بالتوقّع
            preds.append(pred)

        return preds

    def _forecast_statistical(self, horizon_days: int) -> list:
        """
        طريقة إحصائية بسيطة للداتا القليلة:
        متوسط آخر 14 يوم + اتجاه خطي مبسّط.
        """
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

def calc_demand_level(total_units: float, horizon_days: int) -> str:
    """
    بيحسب مستوى الطلب بناءً على متوسط الاستهلاك اليومي:
      Low    → أقل من 50 وحدة/يوم
      Medium → من 50 لـ 100 وحدة/يوم
      High   → أكتر من 100 وحدة/يوم
    """
    daily_avg = total_units / max(1, horizon_days)
    if daily_avg < 50:
        return "Low"
    elif daily_avg <= 100:
        return "Medium"
    else:
        return "High"

# ─── Main Endpoint ────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    يستقبل من الـ .NET:
      - inventory_logs → سجلات حركة المخزون (جدول InventoryLogs)
      - blood_bags     → أكياس الدم المتاحة (جدول BloodBags، Status=Available فقط)
      - horizon_days   → الفترة المطلوب التنبؤ بها (افتراضي 7 أيام)

    ويرجع:
      - التنبؤ بالاستهلاك لكل فصيلة
      - دقة الموديل (أو None لو الداتا قليلة)
      - حالة المخزون وكفايته
    """
    horizon  = request.horizon_days or 7
    stock    = calc_current_stock(request.blood_bags)
    warnings = []

    # اجمع الفصايل الموجودة في السجلات
    blood_types_in_logs = list({log.blood_type.strip() for log in request.inventory_logs})

    predictions = []
    for bt in blood_types_in_logs:

        dates, values = build_daily_consumption(request.inventory_logs, bt)
        if not values:
            continue

        # ── تدريب الموديل وحساب الدقة ──────────────────────────────────
        fc      = BloodTypeForecaster(blood_type=bt).fit(dates, values)
        per_day = fc.forecast(horizon)
        total   = round(sum(per_day), 2)

        # ── حساب حالة المخزون ──────────────────────────────────────────
        cur_stock = float(stock.get(bt, 0))
        avg_daily = total / horizon
        days_cov  = round(cur_stock / avg_daily, 1) if avg_daily > 0 else None
        shortage  = cur_stock < total

        if fc.method == "statistical_fallback":
            warnings.append(
                f"فصيلة {bt}: الداتا أقل من {MIN_POINTS} يوم — "
                f"اُستخدمت طريقة إحصائية والدقة غير متاحة."
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
            days_of_coverage=days_cov,
            shortage_expected=shortage,
        ))

    # رتب تنازلياً حسب الاستهلاك (الأكثر احتياجاً الأول)
    predictions.sort(key=lambda x: x.predicted_total, reverse=True)

    # ── مقاييس كلية ────────────────────────────────────────────────────
    total_units = sum(p.predicted_total for p in predictions)
    accs        = [p.accuracy_percent for p in predictions
                   if p.accuracy_percent is not None]
    overall_acc = round(sum(accs) / len(accs), 2) if accs else None

    return PredictResponse(
        hospital_id=request.hospital_id,
        horizon_days=horizon,
        demand_level=calc_demand_level(total_units, horizon),
        total_expected_units=round(total_units, 2),
        overall_accuracy_percent=overall_acc,
        warnings=warnings,
        predictions=predictions,
    )


@app.get("/health")
def health():
    return {"status": "ok", "service": "Blood Demand Prediction AI"}
