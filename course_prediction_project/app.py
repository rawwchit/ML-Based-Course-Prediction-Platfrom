"""
app.py
------
Course Prediction Platform — Streamlit Web Application

Loads pre-trained artifacts (model.pkl, scaler.pkl, label_encoder.pkl)
and provides an interface for students to receive a course recommendation
based on academic performance, skills, and achievements.

Run:
    streamlit run app.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────
# Page config (must be the first Streamlit call)
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Course Prediction Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "Course Prediction Platform — ML-Based Course Recommender",
    },
)

# ──────────────────────────────────────────────────────────────
# Design System — White + Dark Charcoal
#
# Background:  #ffffff (white)
# Surface:     #ffffff (white cards)
# Border:      #e5e7eb (light gray)
# Accent:      #1e293b (dark charcoal)  ← single dark contrast
# Text:        #1e293b
# Text-muted:  #6b7280
# ──────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    -webkit-font-smoothing: antialiased;
}

/* ── App background ── */
.stApp {
    background-color: #ffffff !important;
    color: #1e293b !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #f9fafb !important;
    border-right: 1px solid #e5e7eb !important;
}

/* ── Main container ── */
.block-container {
    padding-top: 2.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1100px;
}

/* ── Page header ── */
.page-header {
    border-bottom: 1px solid #e5e7eb;
    padding-bottom: 1.2rem;
    margin-bottom: 2rem;
}
.page-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    font-weight: 400;
    color: #1e293b;
    letter-spacing: -0.02em;
    margin: 0 0 0.25rem 0;
}
.page-subtitle {
    font-size: 0.88rem;
    color: #6b7280;
    margin: 0;
    line-height: 1.5;
}

/* ── Form sections ── */
.form-section {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 1.4rem 1.6rem 1rem;
    margin-bottom: 1.2rem;
}
.form-section-title {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #1e293b;
    margin: 0 0 1rem 0;
    padding-bottom: 0.55rem;
    border-bottom: 1px solid #e5e7eb;
}

/* ── Result panel ── */
.result-panel {
    background: #ffffff;
    border: 1px solid #1e293b;
    border-left: 4px solid #1e293b;
    border-radius: 8px;
    padding: 2rem 2.2rem;
    margin-top: 1.6rem;
}
.result-eyebrow {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #1e293b;
    margin: 0 0 0.4rem 0;
}
.result-name-line {
    font-size: 0.88rem;
    color: #6b7280;
    margin: 0 0 0.7rem 0;
}
.result-course {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    font-weight: 400;
    color: #1e293b;
    margin: 0 0 0.7rem 0;
    letter-spacing: -0.01em;
}
.result-desc {
    font-size: 0.86rem;
    color: #6b7280;
    line-height: 1.65;
    margin: 0 0 1rem 0;
}
.career-chip {
    display: inline-block;
    background: #f3f4f6;
    border: 1px solid #d1d5db;
    border-radius: 4px;
    padding: 0.2rem 0.65rem;
    font-size: 0.76rem;
    font-weight: 500;
    color: #1e293b;
    margin: 0.2rem 0.2rem 0 0;
}

/* ── Confidence section ── */
.confidence-title {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6b7280;
    margin: 1.8rem 0 0.8rem 0;
}
.conf-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.5rem;
}
.conf-label {
    font-size: 0.82rem;
    color: #6b7280;
    min-width: 195px;
}
.conf-label.best {
    color: #1e293b;
    font-weight: 600;
}
.conf-bar-wrap {
    flex: 1;
    background: #e5e7eb;
    border-radius: 3px;
    height: 6px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 3px;
    background: #1e293b;
    transition: width 0.4s ease;
}
.conf-bar-fill.dim {
    background: #d1d5db;
}
.conf-pct {
    font-size: 0.8rem;
    color: #6b7280;
    min-width: 42px;
    text-align: right;
}
.conf-pct.best {
    color: #1e293b;
    font-weight: 600;
}

/* ── Streamlit input labels ── */
div[data-testid="stNumberInput"] label,
div[data-testid="stSlider"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stRadio"] label {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #374151 !important;
}

/* ── Slider thumb ── */
div[data-testid="stSlider"] [role="slider"] {
    background-color: #1e293b !important;
    border-color: #1e293b !important;
}

/* ── Submit button ── */
div.stButton > button,
button[kind="formSubmit"] {
    background: #1e293b !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.6rem 2rem !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    transition: background 0.2s ease, box-shadow 0.2s ease !important;
    width: 100% !important;
}
div.stButton > button:hover,
button[kind="formSubmit"]:hover {
    background: #0f172a !important;
    box-shadow: 0 3px 12px rgba(30, 41, 59, 0.25) !important;
}

/* ── Progress bars ── */
.stProgress > div > div {
    background: #1e293b !important;
}

/* ── Divider ── */
hr { border-color: #e5e7eb; margin: 1.2rem 0; }

/* ── Expander ── */
div[data-testid="stExpander"] summary {
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: #374151;
}

/* ── Table styling ── */
.stTable th {
    background-color: #f9fafb !important;
    color: #1e293b !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
.stTable td {
    background-color: #ffffff !important;
    color: #374151 !important;
    font-size: 0.84rem !important;
}

/* ── Footer ── */
.app-footer {
    margin-top: 2.5rem;
    padding-top: 0.8rem;
    border-top: 1px solid #e5e7eb;
    font-size: 0.74rem;
    color: #9ca3af;
    text-align: center;
    letter-spacing: 0.03em;
}

/* ── Sidebar styling ── */
.sidebar-brand {
    font-family: 'DM Serif Display', serif;
    font-size: 1.2rem;
    color: #1e293b;
    margin: 0 0 0.2rem 0;
}
.sidebar-tagline {
    font-size: 0.76rem;
    color: #6b7280;
    margin: 0;
}
.sidebar-status {
    display: inline-block;
    background: #f3f4f6;
    border: 1px solid #d1d5db;
    color: #1e293b;
    border-radius: 4px;
    padding: 0.18rem 0.6rem;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin: 0.7rem 0;
}
.sidebar-status.error {
    background: #fef2f2;
    border-color: #fca5a5;
    color: #991b1b;
}
.sidebar-section-label {
    font-size: 0.66rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #9ca3af;
    margin: 1rem 0 0.4rem 0;
}
.sidebar-class-item {
    font-size: 0.8rem;
    color: #374151;
    padding: 0.15rem 0;
    border-left: 2px solid #e5e7eb;
    padding-left: 0.55rem;
    margin-bottom: 0.25rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Artifact loading
# ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource(show_spinner=False)
def load_artifacts(base_dir: str):
    """Load model, scaler, and label encoder from the project directory."""
    paths = {
        "model":   os.path.join(base_dir, "model.pkl"),
        "scaler":  os.path.join(base_dir, "scaler.pkl"),
        "encoder": os.path.join(base_dir, "label_encoder.pkl"),
    }
    missing = [k for k, v in paths.items() if not os.path.exists(v)]
    if missing:
        return None, None, None, missing

    with open(paths["model"],   "rb") as f: model  = pickle.load(f)
    with open(paths["scaler"],  "rb") as f: scaler = pickle.load(f)
    with open(paths["encoder"], "rb") as f: le     = pickle.load(f)
    return model, scaler, le, []


model, scaler, le, missing_files = load_artifacts(BASE_DIR)

# ──────────────────────────────────────────────────────────────
# Course metadata
# ──────────────────────────────────────────────────────────────
COURSE_INFO = {
    "Computer Science": {
        "desc": (
            "Focuses on algorithms, data structures, software engineering, "
            "artificial intelligence, and systems design. Well-suited for "
            "students with strong logical reasoning and interest in technology."
        ),
        "careers": [
            "Software Engineer", "AI / ML Engineer",
            "Cybersecurity Analyst", "Cloud Architect",
        ],
    },
    "Data Science": {
        "desc": (
            "Combines statistical analysis, machine learning, and data visualization "
            "to extract actionable insights from large datasets. Ideal for analytical "
            "and research-oriented profiles."
        ),
        "careers": [
            "Data Scientist", "Business Intelligence Analyst",
            "ML Engineer", "Research Scientist",
        ],
    },
    "Mechanical Engineering": {
        "desc": (
            "Covers the design, analysis, and manufacturing of mechanical systems "
            "across automotive, aerospace, and industrial domains."
        ),
        "careers": [
            "Mechanical Engineer", "Product Design Engineer",
            "Automotive Engineer", "R&D Specialist",
        ],
    },
    "Civil Engineering": {
        "desc": (
            "Addresses the planning, design, and construction of infrastructure such as "
            "bridges, highways, and urban systems."
        ),
        "careers": [
            "Civil Engineer", "Urban Planner",
            "Structural Engineer", "Project Manager",
        ],
    },
    "Electrical Engineering": {
        "desc": (
            "Encompasses electronics, power systems, control theory, and embedded systems. "
            "Suited for students with high scores in physics and mathematics."
        ),
        "careers": [
            "Electrical Engineer", "Embedded Systems Developer",
            "Power Systems Engineer", "IoT Engineer",
        ],
    },
    "Business": {
        "desc": (
            "Covers management, marketing, finance, and organizational strategy. "
            "Best suited for students demonstrating strong communication and leadership skills."
        ),
        "careers": [
            "Business Analyst", "Marketing Manager",
            "Finance Analyst", "Management Consultant",
        ],
    },
    "Arts & Design": {
        "desc": (
            "Develops expertise in visual communication, user experience design, "
            "illustration, and creative direction."
        ),
        "careers": [
            "UX / UI Designer", "Graphic Designer",
            "Art Director", "Creative Strategist",
        ],
    },
}

# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<p class="sidebar-brand">Course Prediction Platform</p>'
        '<p class="sidebar-tagline">ML-based academic course recommender</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    if model is not None:
        model_name = type(model).__name__
        st.markdown(
            '<span class="sidebar-status">Model Active</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p class="sidebar-tagline" style="margin-top:0.3rem;">'
            f'Algorithm: <strong style="color:#1e293b">{model_name}</strong></p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="sidebar-section-label">Available Courses</p>',
            unsafe_allow_html=True,
        )
        for cls in le.classes_:
            st.markdown(
                f'<div class="sidebar-class-item">{cls}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<span class="sidebar-status error">Model Unavailable</span>',
            unsafe_allow_html=True,
        )
        st.markdown("**Missing files:**")
        for f in missing_files:
            st.markdown(f"- `{f}.pkl`")
        st.info(
            "Run `python train_model.py` to generate the required artifacts."
        )

    st.divider()
    st.markdown(
        '<p style="font-size:0.7rem;color:#94a3b8;letter-spacing:0.03em;">'
        "Built with Streamlit and scikit-learn<br>"
        "For academic and demonstration use"
        "</p>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────
# Page Header
# ──────────────────────────────────────────────────────────────
st.markdown(
    '<div class="page-header">'
    '<h1 class="page-title">Course Prediction Platform</h1>'
    '<p class="page-subtitle">'
    "Complete the form below to receive a machine learning-based course "
    "recommendation tailored to your academic profile, skills, and achievements."
    "</p>"
    "</div>",
    unsafe_allow_html=True,
)

if model is None:
    st.error(
        "The prediction model could not be loaded. "
        "Please run `python train_model.py` and refresh this page.",
    )
    st.stop()

# ──────────────────────────────────────────────────────────────
# Input Form
# ──────────────────────────────────────────────────────────────
with st.form("counselling_form", clear_on_submit=False):

    # 1 — Personal Details
    st.markdown(
        '<div class="form-section">'
        '<p class="form-section-title">Personal Details</p>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([3, 1.5, 1.5])
    with c1:
        name = st.text_input("Full Name", placeholder="Enter your full name")
    with c2:
        age = st.number_input("Age", min_value=14, max_value=25, value=17, step=1)
    with c3:
        gender = st.selectbox("Gender", ["Male", "Female", "Other / Prefer not to say"])
    st.markdown("</div>", unsafe_allow_html=True)

    # 2 — Academic Scores
    st.markdown(
        '<div class="form-section">'
        '<p class="form-section-title">Academic Scores (0 - 100)</p>',
        unsafe_allow_html=True,
    )
    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        math_score = st.number_input(
            "Mathematics", min_value=0, max_value=100, value=70, key="math"
        )
    with s2:
        physics_score = st.number_input(
            "Physics", min_value=0, max_value=100, value=65, key="phys"
        )
    with s3:
        chemistry_score = st.number_input(
            "Chemistry", min_value=0, max_value=100, value=60, key="chem"
        )
    with s4:
        biology_score = st.number_input(
            "Biology", min_value=0, max_value=100, value=60, key="bio"
        )
    with s5:
        english_score = st.number_input(
            "English", min_value=0, max_value=100, value=70, key="eng"
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # 3 — Skills & Interests
    st.markdown(
        '<div class="form-section">'
        '<p class="form-section-title">Skills and Interests (1 - 10)</p>',
        unsafe_allow_html=True,
    )
    r1, r2, r3 = st.columns(3)
    with r1:
        programming_interest = st.slider(
            "Programming Interest", 1, 10, 5,
            help="Aptitude and enthusiasm for programming and software development",
        )
    with r2:
        analytical_skills = st.slider(
            "Analytical Skills", 1, 10, 5,
            help="Logical reasoning, problem-solving, and quantitative thinking",
        )
    with r3:
        creativity_level = st.slider(
            "Creativity Level", 1, 10, 5,
            help="Ability to generate original ideas, visual thinking, design sensibility",
        )
    r4, r5, _ = st.columns(3)
    with r4:
        communication_skills = st.slider(
            "Communication Skills", 1, 10, 5,
            help="Written and verbal communication, presentation ability",
        )
    with r5:
        leadership_skills = st.slider(
            "Leadership Skills", 1, 10, 5,
            help="Team management, initiative, and organizational ability",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # 4 — Achievements
    st.markdown(
        '<div class="form-section">'
        '<p class="form-section-title">Achievements</p>',
        unsafe_allow_html=True,
    )
    a1, a2, a3 = st.columns(3)
    with a1:
        sports_radio = st.radio(
            "Participated in Inter-School / State Sports?",
            ["Yes", "No"], horizontal=True, index=1,
        )
    with a2:
        olympiad_radio = st.radio(
            "Participated in Academic Olympiad?",
            ["Yes", "No"], horizontal=True, index=1,
        )
    with a3:
        projects_done = st.number_input(
            "Independent Projects Completed (0 - 5)",
            min_value=0, max_value=5, value=1,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Submit
    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Generate Recommendation")

# ──────────────────────────────────────────────────────────────
# Prediction & Result
# ──────────────────────────────────────────────────────────────
if submitted:
    sports   = 1 if sports_radio   == "Yes" else 0
    olympiad = 1 if olympiad_radio == "Yes" else 0

    feature_vector = np.array([[
        math_score, physics_score, chemistry_score, biology_score, english_score,
        programming_interest, analytical_skills, creativity_level,
        communication_skills, leadership_skills,
        sports, olympiad, projects_done,
    ]])

    features_scaled = scaler.transform(feature_vector)
    pred_encoded    = model.predict(features_scaled)[0]
    pred_course     = le.inverse_transform([pred_encoded])[0]
    proba           = model.predict_proba(features_scaled)[0]

    proba_df = (
        pd.DataFrame({"Course": le.classes_, "Confidence": proba})
        .sort_values("Confidence", ascending=False)
        .reset_index(drop=True)
    )

    info = COURSE_INFO.get(pred_course, {"desc": "", "careers": []})

    # Result panel
    greeting = (
        f"Recommendation for {name.strip()} — " if name.strip() else "Recommendation — "
    )

    result_html = (
        '<div class="result-panel">'
        '<p class="result-eyebrow">Recommended Course</p>'
        f'<p class="result-name-line">{greeting}based on your submitted profile</p>'
        f'<p class="result-course">{pred_course}</p>'
        f'<p class="result-desc">{info["desc"]}</p>'
    )
    if info.get("careers"):
        chips = " ".join(
            f'<span class="career-chip">{c}</span>' for c in info["careers"]
        )
        result_html += f'<div style="margin-top:0.3rem;">{chips}</div>'
    result_html += "</div>"

    st.markdown(result_html, unsafe_allow_html=True)

    # Confidence breakdown
    st.markdown(
        '<p class="confidence-title">Prediction Confidence — All Courses</p>',
        unsafe_allow_html=True,
    )

    for _, row in proba_df.iterrows():
        is_best = row["Course"] == pred_course
        pct     = row["Confidence"] * 100
        bar_pct = f"{pct:.1f}"
        lbl_cls = "conf-label best" if is_best else "conf-label"
        bar_cls = "conf-bar-fill"   if is_best else "conf-bar-fill dim"
        pct_cls = "conf-pct best"   if is_best else "conf-pct"

        st.markdown(
            f'<div class="conf-row">'
            f'<span class="{lbl_cls}">{row["Course"]}</span>'
            f'<div class="conf-bar-wrap">'
            f'<div class="{bar_cls}" style="width:{bar_pct}%"></div>'
            f'</div>'
            f'<span class="{pct_cls}">{bar_pct}%</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Input summary
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("View Submitted Profile Summary"):
        summary = pd.DataFrame({
            "Field": [
                "Full Name", "Age", "Gender",
                "Mathematics Score", "Physics Score", "Chemistry Score",
                "Biology Score", "English Score",
                "Programming Interest", "Analytical Skills",
                "Creativity Level", "Communication Skills", "Leadership Skills",
                "Sports Participation", "Olympiad Participation",
                "Projects Completed",
            ],
            "Value": [
                str(name or "—"), str(age), str(gender),
                str(math_score), str(physics_score), str(chemistry_score),
                str(biology_score), str(english_score),
                str(programming_interest), str(analytical_skills),
                str(creativity_level), str(communication_skills),
                str(leadership_skills),
                "Yes" if sports else "No",
                "Yes" if olympiad else "No",
                str(projects_done),
            ],
        })
        st.table(summary)

# ──────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────
st.markdown(
    '<div class="app-footer">'
    "Course Prediction Platform &nbsp;|&nbsp; "
    "Machine Learning-Based Course Predictor"
    "</div>",
    unsafe_allow_html=True,
)
