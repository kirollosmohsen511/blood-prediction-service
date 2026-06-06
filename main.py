"""
Blood Demand Prediction AI Service
=====================================
بيتنبأ باستهلاك الدم لكل فصيلة بناءً على تاريخ أكياس الدم.

مصدر البيانات الوحيد: جدول BloodBags
  - الأكياس المتاحة  (Status = Available) → المخزون الحالي
  - الأكياس المستخدمة (Status = Used)      → تاريخ الاستهلاك للموديل

الموديل: RandomForest Regressor
التقييم: Walk-Forward Validation (TimeSeriesSplit - 3 folds)
الدقة:   MAE / RMSE / MAPE / Accuracy%

تشغيل:
  pip install -r requirements.txt
  uvicorn main:app --reload --port 8001
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Union
import math
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

app = FastAPI(title="Blood Demand Prediction AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Constants ────────────────────────────────────────────────────────────────

LAGS         = [1, 2, 3, 7]   # أيام اللي بنرجع ليها كـ features
ROLL_WINDOW  = 7               # نافذة المتوسط المتحرك
MIN_POINTS   = 35              # أقل عدد أيام عشان ندرّب موديل ML بأمانة
N_SPLITS     = 3               # عدد الـ folds في Walk-Forward Validation
RANDOM_STATE = 42

# قيم enum البتاع BloodBagStatus (عدّلها حسب الـ enum عندك في الـ .NET)
AVAILABLE_STATUS = 0   # BloodBagStatus.Available
USED_STATUS      = 1   # BloodBagStatus.Used

# تحويل BloodType enum (int) لاسم مقروء
# عدّل الترتيب حسب الـ BloodType enum في الـ .NET بتاعك
BLOOD_TYPE_LABELS = {
    0: "A+",  1: "A-",  2: "B+",  3: "B-",
    4: "AB+", 5: "AB-", 6: "O+",  7: "O-",
}

# ─── Date Parser ──────────────────────────────────────────────────────────────

def parse_dt(value: str) -> datetime:
    """
    بتحوّل أي صيغة تاريخ جاية من SQL Server لـ datetime بدون timezone.
    بتتعامل مع:
      - "2026-01-15T10:00:00"           (بسيط)
      - "2026-01-15T10:00:00Z"          (UTC)
      - "2026-01-15T10:00:00.1234567"   (SQL Server DateTime2)
      - "2026-01-15T10:00:00+03:00"     (timezone offset)
    """
    ts = pd.to_datetime(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.floor("s").to_pydatetime()

# ─── Blood Type Label ─────────────────────────────────────────────────────────

def label_blood_type(value) -> str:
    """بتحوّل قيمة BloodType من int لاسم مقروء. لو string بترجعها زي ما هي."""
    try:
        return BLOOD_TYPE_LABELS.get(int(value), str(value))
    except (ValueError, TypeError):
        return str(value)

# ─── Models ───────────────────────────────────────────────────────────────────

class BloodBagItem(BaseModel):
    blood_type:  Union[int, str]        # int (enum من الداتابيز) أو str ("A+")
    status:      int                    # 0=Available / 1=Used / 2=Expired ...
    created_at:  str                    # ISO datetime — تاريخ دخول الكيس
    expiry_date: Optional[str] = None  # ISO datetime أو None

class PredictRequest(BaseModel):
    hospital_id:  int
    blood_bags:   List[BloodBagItem]   # كل الأكياس (كل الـ statuses)
    horizon_days: Optional[int] = 7    # الفترة المطلوب التنبؤ بها

class BloodTypePrediction(BaseModel):
    blood_type:        str
    method:            str              # ml_random_forest | statistical_fallback
    predicted_total:   float            # إجمالي الاستهلاك المتوقّع خلال الفترة
    predicted_per_day: List[float]      # توزيع يومي
    accuracy_percent:  Optional[float]  # None لو الداتا قليلة
    mae:               Optional[float]
    rmse:              Optional[float]
    mape:              Optional[float]
    current_stock:     float            # عدد الأكياس المتاحة الصالحة
    units_required:    float            # الوحدات المطلوب طلبها (0 لو المخزون كافي)
    units_surplus:     float            # الوحدات الزيادة (0 لو في عجز)
    days_of_coverage:  Optional[float]  # المخزون يكفي كام يوم
    shortage_expected: bool             # هل المخزون أقل من المتوقّع؟

class PredictResponse(BaseModel):
    hospital_id:              int
    horizon_days:             int
    demand_level:             str              # Low | Medium | High
    total_expected_units:     float            # مجموع الاستهلاك المتوقّع لكل الفصايل
    total_units_required:     float            # مجموع الوحدات المطلوب طلبها
    overall_accuracy_percent: Optional[float]
    warnings:                 List[str]
    predictions:              List[BloodTypePrediction]

# ─── Current Stock from BloodBags ─────────────────────────────────────────────

def calc_current_stock(blood_bags: List[BloodBagItem]) -> dict:
    """
    بيحسب المخزون الحالي من الأكياس المتاحة (Status = Available).
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
                if parse_dt(bag.expiry_date) <= now:
                    continue
            except Exception:
                pass

        bt = label_blood_type(bag.blood_type)
        stock[bt] = stock.get(bt, 0) + 1

    return stock

# ─── Consumption Series from Used Bags ────────────────────────────────────────

def build_daily_consumption(blood_bags: List[BloodBagItem], blood_type: str):
    """
    بيبني سلسلة استهلاك يومية من الأكياس المستخدمة (Status = Used).
    بيستخدم created_at كتاريخ الكيس.
    الأيام اللي مفيهاش استخدام بتتملي بصفر عشان السلسلة تفضل متصلة.
    """
    daily = {}

    for bag in blood_bags:
        if label_blood_type(bag.blood_type) != blood_type:
            continue
        if bag.status != USED_STATUS:
            continue
        try:
            day = parse_dt(bag.created_at).date()
        except Exception:
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
    MAPE بيتحسب على الأيام اللي الاستهلاك فيها > 0 بس.
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
    blood_type:  str
    model:       object = None
    method:      str    = "ml_random_forest"
    metrics:     dict   = field(default_factory=dict)
    last_values: list   = field(default_factory=list)
    last_dates:  list   = field(default_factory=list)

    def fit(self, dates: list, values: list) -> "BloodTypeForecaster":
        """
        بيدرّب الموديل ويحسب الدقة بطريقة Walk-Forward Validation.

        Walk-Forward = بنختبر الموديل على 3 فترات زمنية مختلفة وناخد
        متوسط الدقة — أصدق بكتير من اختبار واحد بس (Single Holdout).

        Fold 1: [train ████████] [test ██]
        Fold 2: [train ████████████] [test ██]
        Fold 3: [train ████████████████] [test ██]
                                   ↓
                        average accuracy = النتيجة النهائية
        """
        self.last_dates, self.last_values = dates, values

        if len(values) < MIN_POINTS:
            # داتا قليلة → statistical fallback بدون accuracy مفبركة
            self.method  = "statistical_fallback"
            self.model   = None
            self.metrics = {"mae": None, "rmse": None,
                            "mape": None, "accuracy_percent": None}
            return self

        X, y = _build_xy(values, dates)

        # ── Walk-Forward Validation ────────────────────────────────────────────
        test_size    = max(5, len(X) // 7)
        n_splits     = max(1, min(N_SPLITS, (len(X) - test_size) // test_size))
        tscv         = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        fold_metrics = []

        for train_idx, test_idx in tscv.split(X):
            if len(train_idx) < 10:
                continue
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y[train_idx],       y[test_idx]
            fold_model = RandomForestRegressor(
                n_estimators=200, min_samples_leaf=2, random_state=RANDOM_STATE
            )
            fold_model.fit(X_tr, y_tr)
            fold_metrics.append(_calc_metrics(y_te, fold_model.predict(X_te)))

        # متوسط الدقة على كل الـ folds
        if fold_metrics:
            def avg(key):
                vals = [m[key] for m in fold_metrics if m[key] is not None]
                return round(sum(vals) / len(vals), 3) if vals else None
            self.metrics = {
                "mae":              avg("mae"),
                "rmse":             avg("rmse"),
                "mape":             avg("mape"),
                "accuracy_percent": avg("accuracy_percent"),
            }
        else:
            self.metrics = {"mae": None, "rmse": None,
                            "mape": None, "accuracy_percent": None}

        # ── موديل نهائي على كل الداتا للتنبؤ ──────────────────────────────────
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
            values.append(0.0)
            i    = len(values) - 1
            row  = _feature_row(values, dates, i)
            pred = max(0.0, float(self.model.predict(pd.DataFrame([row]))[0]))
            values[-1] = pred
            preds.append(pred)
        return preds

    def _forecast_statistical(self, horizon_days: int) -> list:
        """طريقة إحصائية بسيطة للداتا القليلة: متوسط + اتجاه خطي."""
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
      - blood_bags  → كل أكياس الدم بتاعة المستشفى (كل الـ statuses)
      - horizon_days → الفترة المطلوب التنبؤ بها (افتراضي 7 أيام)

    Python بيقسّم الأكياس:
      - Status = Available → المخزون الحالي (بعد استبعاد المنتهية)
      - Status = Used      → تاريخ الاستهلاك لتدريب الموديل
    """
    horizon  = request.horizon_days or 7
    stock    = calc_current_stock(request.blood_bags)
    warnings = []

    blood_types = list({
        label_blood_type(bag.blood_type)
        for bag in request.blood_bags
        if bag.status == USED_STATUS
    })

    predictions = []
    for bt in blood_types:

        dates, values = build_daily_consumption(request.blood_bags, bt)
        if not values:
            continue

        # ── تدريب الموديل بـ Walk-Forward Validation ──────────────────────────
        fc      = BloodTypeForecaster(blood_type=bt).fit(dates, values)
        per_day = fc.forecast(horizon)
        total   = round(sum(per_day), 2)

        # ── حساب حالة المخزون ──────────────────────────────────────────────────
        cur_stock      = float(stock.get(bt, 0))
        avg_daily      = total / horizon
        days_cov       = round(cur_stock / avg_daily, 1) if avg_daily > 0 else None
        shortage       = cur_stock < total
        units_required = round(max(0.0, total - cur_stock), 2)
        units_surplus  = round(max(0.0, cur_stock - total), 2)

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
            units_required=units_required,
            units_surplus=units_surplus,
            days_of_coverage=days_cov,
            shortage_expected=shortage,
        ))

    # رتب تنازلياً حسب الوحدات المطلوبة (الأكثر احتياجاً الأول)
    predictions.sort(key=lambda x: x.units_required, reverse=True)

    total_units = sum(p.predicted_total for p in predictions)
    accs        = [p.accuracy_percent for p in predictions
                   if p.accuracy_percent is not None]
    overall_acc = round(sum(accs) / len(accs), 2) if accs else None

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
