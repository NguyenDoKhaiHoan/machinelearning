from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model", "best_model.joblib"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "model", "feature_columns.joblib"))


DATA_PATH = os.path.join(BASE_DIR, "train_final.csv")  
train_df = pd.read_csv(DATA_PATH)


COURSE_MAP = {
    33: "Biofuel Production Technologies",
    171: "Animation and Multimedia Design",
    8014: "Social Service (evening attendance)",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equinculture",
    9147: "Management",
    9238: "Social Service",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication",
    9853: "Basic Education",
    9991: "Management (evening attendance)"
}

MARITAL_STATUS_MAP = {
    1: "single",
    2: "married",
    3: "widower",
    4: "divorced",
    5: "facto union legally separated"
}


NUMERIC_COLS = [
    "Age at enrollment",
    "Admission grade",
    "Previous qualification (grade)",
    "Application order",
    "Application mode",
]


CATEGORICAL_COLS = [
     "Attendance_mode",
    "Displaced", "Educational special needs", "Debtor",
    "Tuition fees up to date", "Gender", "Scholarship holder",
    "International", "father_occ", "father_qual", "mother_occ",
    "mother_qual", "previous_qual"
]

CLASS_NAME_MAP = {
    0: "Sinh viên có khả năng bỏ học",
    1: "Sinh viên có khả năng tốt nghiệp",
}

options = {}

options["Course"] = COURSE_MAP
options["Marital status"] = MARITAL_STATUS_MAP

for col in CATEGORICAL_COLS:
    if col in train_df.columns:
        vals = train_df[col].dropna().astype(object).unique().tolist()
        try:
            vals = sorted(vals)
        except Exception:
            pass
        options[col] = vals


ranges = {}
for col in NUMERIC_COLS:
    if col in train_df.columns:
        s = pd.to_numeric(train_df[col], errors="coerce").dropna()
        if len(s) > 0:
            mn, mx = float(s.min()), float(s.max())
            step = 1.0 if col in ["Age at enrollment", "Application order"] else 0.1
            ranges[col] = {"min": mn, "max": mx, "step": step}


DEFAULTS = {}
for c in feature_columns:
    if c in train_df.columns:
        try:
            DEFAULTS[c] = train_df[c].dropna().iloc[0]
        except Exception:
            DEFAULTS[c] = ""

if "Course" in feature_columns and len(COURSE_MAP) > 0:
    DEFAULTS["Course"] = list(COURSE_MAP.keys())[0]
if "Marital status" in feature_columns and len(MARITAL_STATUS_MAP) > 0:
    DEFAULTS["Marital status"] = list(MARITAL_STATUS_MAP.keys())[0]

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probs = None

    if request.method == "POST":
        row = {}
        for col in feature_columns:
            val = request.form.get(col, "").strip()
            if val == "" or val is None:
                row[col] = DEFAULTS.get(col, "")
            else:
                row[col] = val

        df = pd.DataFrame([row], columns=feature_columns)

        for c in df.columns:
            if c in train_df.columns and pd.api.types.is_numeric_dtype(train_df[c]):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            else:
                df[c] = df[c].astype(object)

        for c in ["Course", "Marital status"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        pred = model.predict(df)[0]

        result = CLASS_NAME_MAP.get(int(pred), str(pred))

        probs = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[0]
            classes = model.named_steps["model"].classes_
            probs = [
                (CLASS_NAME_MAP.get(int(c), str(c)), float(p))
                for c, p in zip(classes, proba)
            ]
            probs.sort(key=lambda x: x[1], reverse=True)


    return render_template(
        "index.html",
        columns=feature_columns,
        result=result,
        probs=probs,
        options=options,
        ranges=ranges
    )

if __name__ == "__main__":
    app.run(debug=True)
