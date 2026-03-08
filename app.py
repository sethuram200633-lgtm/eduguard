import os
import io
import re
import pickle
import random
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="EduGuard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Data & model loading
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = "data/students.csv"
MODELS_DIR = "models"

_df: Optional[pd.DataFrame] = None
_regressor = None
_classifier = None
_label_encoders = None
_feature_cols: Optional[List[str]] = None


def _load_data():
    global _df
    if os.path.exists(DATA_PATH):
        _df = pd.read_csv(DATA_PATH)
    else:
        _df = pd.DataFrame()


def _load_models():
    global _regressor, _classifier, _label_encoders, _feature_cols
    try:
        with open(f"{MODELS_DIR}/regressor.pkl", "rb") as f:
            _regressor = pickle.load(f)
        with open(f"{MODELS_DIR}/classifier.pkl", "rb") as f:
            _classifier = pickle.load(f)
        with open(f"{MODELS_DIR}/label_encoders.pkl", "rb") as f:
            _label_encoders = pickle.load(f)
        with open(f"{MODELS_DIR}/feature_cols.pkl", "rb") as f:
            _feature_cols = pickle.load(f)
    except FileNotFoundError:
        pass  # Models not trained yet – handled in /predict


@app.on_event("startup")
def startup_event():
    _load_data()
    _load_models()


def get_df() -> pd.DataFrame:
    if _df is None or _df.empty:
        raise HTTPException(status_code=503, detail="No student data loaded. Please upload a CSV.")
    return _df


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build ML features for a row
# ─────────────────────────────────────────────────────────────────────────────
def _build_features(row: pd.Series) -> np.ndarray:
    family_income_enc = _label_encoders["income"].transform([row["family_income"]])[0]
    residence_enc = _label_encoders["residence"].transform([row["residence"]])[0]
    return np.array([[
        row["attendance_pct"],
        row["exam_score"],
        row["distance_km"],
        row["midday_meal"],
        row["sibling_dropout"],
        row["prev_year_score"],
        row["teacher_engagement"],
        row["extracurricular"],
        family_income_enc,
        residence_enc,
    ]])


# ─────────────────────────────────────────────────────────────────────────────
# Intervention & government scheme data
# ─────────────────────────────────────────────────────────────────────────────
INTERVENTIONS = {
    "low_attendance": {
        "problem": "Low Attendance",
        "icon": "📅",
        "priority": 1,
        "actions": [
            "Schedule a home visit by the class teacher",
            "Send attendance alert to parents via SMS/WhatsApp",
            "Connect student with a peer buddy",
        ],
    },
    "low_exam_score": {
        "problem": "Academic Struggle",
        "icon": "📚",
        "priority": 2,
        "actions": [
            "Enroll in peer mentoring / tutoring program",
            "Provide remedial classes after school hours",
            "Assign subject-specific learning materials",
        ],
    },
    "long_distance": {
        "problem": "Long Distance from School",
        "icon": "🚌",
        "priority": 3,
        "actions": [
            "Apply for school transport scheme",
            "Explore residential school options",
            "Connect with district education officer for transport support",
        ],
    },
    "financial_hardship": {
        "problem": "Financial Hardship",
        "icon": "💰",
        "priority": 4,
        "actions": [
            "Apply for government scholarship (NSP/State scheme)",
            "Enroll in Midday Meal Programme",
            "Connect with NGO support groups",
        ],
    },
    "sibling_dropout": {
        "problem": "Sibling Dropout History",
        "icon": "👨‍👩‍👧‍👦",
        "priority": 5,
        "actions": [
            "Family counseling session",
            "Motivational talk by school counselor",
            "Engage parents with school activities",
        ],
    },
    "low_engagement": {
        "problem": "Low Teacher Engagement",
        "icon": "🤝",
        "priority": 6,
        "actions": [
            "One-on-one mentoring with teacher",
            "Enroll in extracurricular activities",
            "Counseling support",
        ],
    },
}

GOV_SCHEMES = [
    {"name": "National Scholarship Portal (NSP)", "type": "Scholarship", "eligibility": "Low income families", "url": "scholarships.gov.in"},
    {"name": "Sarva Shiksha Abhiyan (SSA)", "type": "Education Support", "eligibility": "All students upto 8th grade", "url": "ssa.nic.in"},
    {"name": "Pradhan Mantri Poshan Shakti Nirman", "type": "Midday Meal", "eligibility": "All school students", "url": "pmposhan.education.gov.in"},
    {"name": "National Means-cum-Merit Scholarship (NMMS)", "type": "Scholarship", "eligibility": "Grade 8 students from low-income families", "url": "scholarships.gov.in"},
    {"name": "State Transport Scheme", "type": "Transport", "eligibility": "Students living > 5km from school", "url": "state.education.gov.in"},
    {"name": "Kasturba Gandhi Balika Vidyalaya (KGBV)", "type": "Residential School", "eligibility": "Girls in educationally backward areas", "url": "samagrashiksha.gov.in"},
    {"name": "RTE Act Free Education", "type": "Right to Education", "eligibility": "All students aged 6-14 years", "url": "righttoeducation.in"},
]

# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "EduGuard API is running", "version": "1.0.0"}


@app.get("/api/students")
def list_students(search: Optional[str] = None, risk: Optional[str] = None, grade: Optional[str] = None):
    df = get_df()
    if search:
        df = df[df["name"].str.contains(search, case=False, na=False)]
    if risk and risk != "all":
        df = df[df["risk_label"].str.lower() == risk.lower()]
    if grade and grade != "all":
        df = df[df["grade"].astype(str) == grade]
    cols = ["student_id", "name", "grade", "gender", "age", "attendance_pct",
            "exam_score", "distance_km", "risk_score", "risk_label", "parent_contact"]
    return df[cols].to_dict(orient="records")


@app.get("/api/students/{student_id}")
def get_student(student_id: str):
    df = get_df()
    row = df[df["student_id"] == student_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Student not found")
    return row.iloc[0].to_dict()


@app.get("/api/predict/{student_id}")
def predict_risk(student_id: str):
    df = get_df()
    row = df[df["student_id"] == student_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Student not found")
    s = row.iloc[0]

    if _regressor is None:
        # Fallback: use precomputed score
        score = float(s["risk_score"])
    else:
        features = _build_features(s)
        score = float(np.clip(_regressor.predict(features)[0], 0, 100))

    if score <= 40:
        level = "Low"
        color = "green"
    elif score <= 70:
        level = "Medium"
        color = "yellow"
    else:
        level = "High"
        color = "red"

    # XAI factors
    factors = []
    if s["attendance_pct"] < 70:
        factors.append({"factor": "Low Attendance", "value": f"{s['attendance_pct']}%", "severity": "high", "icon": "📅"})
    if s["exam_score"] < 50:
        factors.append({"factor": "Poor Exam Scores", "value": f"{s['exam_score']}/100", "severity": "high", "icon": "📝"})
    if s["distance_km"] > 8:
        factors.append({"factor": "Long Distance from School", "value": f"{s['distance_km']} km", "severity": "medium", "icon": "🚌"})
    if s["sibling_dropout"] == 1:
        factors.append({"factor": "Sibling Dropout History", "value": "Yes", "severity": "medium", "icon": "👨‍👩‍👧‍👦"})
    if s["family_income"] == "Low":
        factors.append({"factor": "Low Family Income", "value": "Low", "severity": "medium", "icon": "💰"})
    if s["prev_year_score"] < 50:
        factors.append({"factor": "Weak Previous Year Performance", "value": f"{s['prev_year_score']}/100", "severity": "medium", "icon": "📊"})
    if s["teacher_engagement"] < 4:
        factors.append({"factor": "Low Teacher Engagement", "value": f"{s['teacher_engagement']}/10", "severity": "low", "icon": "🤝"})
    if not s["midday_meal"]:
        factors.append({"factor": "Not Enrolled in Midday Meal", "value": "No", "severity": "low", "icon": "🍱"})

    # Attendance trend (last 6 months, simulated)
    base = float(s["attendance_pct"])
    random.seed(int(s["student_id"][3:]))
    trend = [round(base + random.uniform(-10, 10), 1) for _ in range(6)]
    trend = [max(0, min(100, v)) for v in trend]

    return {
        "student_id": student_id,
        "name": s["name"],
        "risk_score": round(score, 1),
        "risk_level": level,
        "risk_color": color,
        "xai_factors": factors,
        "attendance_trend": trend,
    }


@app.get("/api/interventions/{student_id}")
def get_interventions(student_id: str):
    df = get_df()
    row = df[df["student_id"] == student_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Student not found")
    s = row.iloc[0]

    suggestions = []
    if s["attendance_pct"] < 70:
        suggestions.append(INTERVENTIONS["low_attendance"])
    if s["exam_score"] < 50:
        suggestions.append(INTERVENTIONS["low_exam_score"])
    if s["distance_km"] > 8:
        suggestions.append(INTERVENTIONS["long_distance"])
    if s["family_income"] == "Low":
        suggestions.append(INTERVENTIONS["financial_hardship"])
    if s["sibling_dropout"] == 1:
        suggestions.append(INTERVENTIONS["sibling_dropout"])
    if s["teacher_engagement"] < 4:
        suggestions.append(INTERVENTIONS["low_engagement"])

    # Match relevant government schemes
    schemes = []
    if s["family_income"] == "Low":
        schemes += [GOV_SCHEMES[0], GOV_SCHEMES[3], GOV_SCHEMES[6]]
    if s["distance_km"] > 5:
        schemes.append(GOV_SCHEMES[4])
    if not s["midday_meal"]:
        schemes.append(GOV_SCHEMES[2])
    schemes.append(GOV_SCHEMES[1])
    # Deduplicate
    seen = set()
    unique_schemes = []
    for sc in schemes:
        if sc["name"] not in seen:
            seen.add(sc["name"])
            unique_schemes.append(sc)

    return {
        "student_id": student_id,
        "name": s["name"],
        "interventions": sorted(suggestions, key=lambda x: x["priority"]),
        "government_schemes": unique_schemes,
    }


@app.get("/api/messages/{student_id}")
def generate_message(student_id: str, language: str = "english", channel: str = "whatsapp"):
    df = get_df()
    row = df[df["student_id"] == student_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Student not found")
    s = row.iloc[0]
    name = s["name"].split()[0]
    risk = s["risk_label"]
    attendance = s["attendance_pct"]

    if language == "hindi":
        if risk == "High":
            msg = (
                f"नमस्ते! {name} के अभिभावक से अनुरोध है कि हम आपके बच्चे की शिक्षा यात्रा में "
                f"आपका सहयोग चाहते हैं। हाल ही में विद्यालय में उपस्थिति कम रही है। "
                f"कृपया एक बैठक हेतु विद्यालय से संपर्क करें। हम मिलकर बेहतर कर सकते हैं। 🙏"
            )
        else:
            msg = (
                f"नमस्ते! {name} के विद्यालय की तरफ से आपको सूचित करना है कि आपका बच्चा "
                f"अच्छा कर रहा है। निरंतर प्रोत्साहन के लिए धन्यवाद! 😊"
            )
    else:  # english
        if risk == "High":
            msg = (
                f"Dear Parent/Guardian of {s['name']},\n\n"
                f"We hope this message finds you well. We are reaching out because we care deeply "
                f"about {name}'s learning journey. Recently, we've noticed that {name} has been "
                f"missing school more often (attendance: {attendance}%). We would love to connect "
                f"with you to understand how we can provide better support together.\n\n"
                f"Kindly contact the class teacher at your earliest convenience. "
                f"We believe in {name}'s potential and are here to help. 🌟\n\n"
                f"Warm regards,\nEduGuard School Support Team"
            )
        elif risk == "Medium":
            msg = (
                f"Dear Parent/Guardian of {s['name']},\n\n"
                f"We wanted to share a quick update on {name}'s progress. "
                f"There is an opportunity to improve attendance and academic performance. "
                f"With a little extra encouragement from home, we are confident {name} can shine! "
                f"Please feel free to reach out to us anytime.\n\n"
                f"Best wishes,\nEduGuard School Support Team"
            )
        else:
            msg = (
                f"Dear Parent/Guardian of {s['name']},\n\n"
                f"Great news! {name} is doing well at school. Thank you for your continued support "
                f"in {name}'s education journey. Keep up the wonderful work! 🎉\n\n"
                f"Warm regards,\nEduGuard School Support Team"
            )

    if channel == "whatsapp":
        msg = msg.replace("\n\n", "\n\n")  # keep linebreaks for WhatsApp
    else:
        # SMS: strip to 160 chars summary
        msg = re.sub(r'\n+', ' ', msg)
        msg = msg[:300] + "..." if len(msg) > 300 else msg

    return {
        "student_id": student_id,
        "name": s["name"],
        "parent_contact": s["parent_contact"],
        "language": language,
        "channel": channel,
        "message": msg,
    }


@app.get("/api/district")
def district_analytics():
    df = get_df()
    schools = [
        "Govt. Primary School Anandpur", "Govt. Middle School Balpur", "Govt. High School Chandpur",
        "Govt. Primary School Devpur", "Govt. Middle School Eklavya Nagar",
        "Govt. High School Fatehpur", "Govt. Primary School Govindpur",
        "Govt. Middle School Haripur", "Govt. High School Indrapur",
        "Govt. Primary School Jaganpur",
    ]
    zones = ["North Zone", "South Zone", "East Zone", "West Zone", "Central Zone"]

    random.seed(10)
    school_data = []
    students_per_school = len(df) // len(schools)
    for i, school in enumerate(schools):
        chunk = df.iloc[i * students_per_school:(i + 1) * students_per_school]
        avg_risk = round(chunk["risk_score"].mean(), 1) if not chunk.empty else random.uniform(30, 80)
        high_count = int((chunk["risk_label"] == "High").sum()) if not chunk.empty else random.randint(0, 5)
        school_data.append({
            "school": school,
            "zone": zones[i % len(zones)],
            "avg_risk": avg_risk,
            "total_students": len(chunk),
            "high_risk_count": high_count,
            "medium_risk_count": int((chunk["risk_label"] == "Medium").sum()) if not chunk.empty else random.randint(2, 8),
            "low_risk_count": int((chunk["risk_label"] == "Low").sum()) if not chunk.empty else random.randint(3, 10),
            "risk_level": "High" if avg_risk > 70 else "Medium" if avg_risk > 40 else "Low",
        })

    overall = {
        "total_students": int(df.shape[0]),
        "high_risk": int((df["risk_label"] == "High").sum()),
        "medium_risk": int((df["risk_label"] == "Medium").sum()),
        "low_risk": int((df["risk_label"] == "Low").sum()),
        "avg_risk_score": round(df["risk_score"].mean(), 1),
    }
    return {"schools": school_data, "overall": overall}


@app.get("/api/outcomes/{student_id}")
def get_outcomes(student_id: str):
    """Simulated 30-day outcome tracker after intervention."""
    df = get_df()
    row = df[df["student_id"] == student_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Student not found")
    s = row.iloc[0]

    random.seed(int(s["student_id"][3:]) + 999)
    base_att = float(s["attendance_pct"])
    base_marks = float(s["exam_score"])

    # Simulate gradual improvement over 30 days
    attendance_trend = []
    marks_trend = []
    engagement_trend = []
    for day in range(1, 31):
        factor = day / 30
        att = round(min(100, base_att + (85 - base_att) * factor * random.uniform(0.7, 1.3)), 1)
        marks = round(min(100, base_marks + (70 - base_marks) * factor * random.uniform(0.5, 1.1)), 1)
        eng = round(min(10, 5 + (8 - 5) * factor * random.uniform(0.5, 1.2)), 1)
        attendance_trend.append({"day": day, "value": att})
        marks_trend.append({"day": day, "value": marks})
        engagement_trend.append({"day": day, "value": eng})

    current_att = attendance_trend[-1]["value"]
    current_marks = marks_trend[-1]["value"]
    status = "Improving" if current_att > base_att + 5 else "Stable"

    return {
        "student_id": student_id,
        "name": s["name"],
        "intervention_start": "2026-02-07",
        "days_tracked": 30,
        "attendance_trend": attendance_trend,
        "marks_trend": marks_trend,
        "engagement_trend": engagement_trend,
        "summary": {
            "initial_attendance": round(base_att, 1),
            "current_attendance": current_att,
            "initial_marks": round(base_marks, 1),
            "current_marks": current_marks,
            "status": status,
        },
    }


@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    global _df
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    contents = await file.read()
    df_new = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    required_cols = {"student_id", "name", "attendance_pct", "exam_score", "risk_score"}
    if not required_cols.issubset(df_new.columns):
        raise HTTPException(
            status_code=400,
            detail=f"CSV missing required columns: {required_cols - set(df_new.columns)}"
        )
    os.makedirs("data", exist_ok=True)
    df_new.to_csv(DATA_PATH, index=False)
    _df = df_new
    return {"message": f"Uploaded {len(df_new)} student records successfully.", "columns": list(df_new.columns)}


@app.get("/api/alerts")
def get_alerts():
    """Return all students with risk_score > 70 as alerts."""
    df = get_df()
    alerts = df[df["risk_score"] > 70][
        ["student_id", "name", "grade", "risk_score", "risk_label", "attendance_pct", "exam_score"]
    ].sort_values("risk_score", ascending=False)
    return alerts.to_dict(orient="records")
