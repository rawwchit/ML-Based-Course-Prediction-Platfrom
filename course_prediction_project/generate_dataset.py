"""
generate_dataset.py
--------------------
Generates a synthetic student dataset for the Career Counselling System.
The data follows realistic rule-based patterns to simulate real-world behavior.

Rules:
    - High math + programming  → Computer Science
    - High math + analytics    → Data Science
    - High physics + math      → Mechanical / Electrical Engineering
    - High chemistry + biology → (not primary targets, but used as noise)
    - High creativity          → Arts & Design
    - High communication + leadership → Business
    - High physics + math (civil weight) → Civil Engineering
"""

import numpy as np
import pandas as pd
import random

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

N = 800  # Number of rows to generate

# ─────────────────────────────────────────────
# Helper: clamp values to a valid range
# ─────────────────────────────────────────────

def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def rand_score(mean, std, lo=0, hi=100):
    """Generate a random academic score around a mean with some noise."""
    return clamp(int(np.random.normal(mean, std)), lo, hi)


def rand_skill(mean, std, lo=1, hi=10):
    """Generate a random skill rating around a mean with some noise."""
    return clamp(round(np.random.normal(mean, std), 1), lo, hi)


# ─────────────────────────────────────────────
# Profile generators for each course
# ─────────────────────────────────────────────

def profile_computer_science():
    return {
        "math_score":             rand_score(80, 10),
        "physics_score":          rand_score(72, 10),
        "chemistry_score":        rand_score(60, 12),
        "biology_score":          rand_score(55, 12),
        "english_score":          rand_score(65, 10),
        "programming_interest":   rand_skill(8.5, 1.0),
        "analytical_skills":      rand_skill(8.0, 1.0),
        "creativity_level":       rand_skill(6.0, 1.5),
        "communication_skills":   rand_skill(6.5, 1.5),
        "leadership_skills":      rand_skill(6.0, 1.5),
        "sports":                 random.choices([0, 1], weights=[0.6, 0.4])[0],
        "olympiad":               random.choices([0, 1], weights=[0.5, 0.5])[0],
        "projects_done":          random.choices(range(6), weights=[0.05, 0.10, 0.20, 0.25, 0.25, 0.15])[0],
        "course":                 "Computer Science",
    }


def profile_data_science():
    return {
        "math_score":             rand_score(82, 9),
        "physics_score":          rand_score(68, 11),
        "chemistry_score":        rand_score(62, 12),
        "biology_score":          rand_score(60, 12),
        "english_score":          rand_score(67, 10),
        "programming_interest":   rand_skill(7.5, 1.2),
        "analytical_skills":      rand_skill(9.0, 0.8),
        "creativity_level":       rand_skill(6.5, 1.5),
        "communication_skills":   rand_skill(7.0, 1.5),
        "leadership_skills":      rand_skill(6.5, 1.5),
        "sports":                 random.choices([0, 1], weights=[0.6, 0.4])[0],
        "olympiad":               random.choices([0, 1], weights=[0.55, 0.45])[0],
        "projects_done":          random.choices(range(6), weights=[0.05, 0.10, 0.20, 0.25, 0.25, 0.15])[0],
        "course":                 "Data Science",
    }


def profile_mechanical_engineering():
    return {
        "math_score":             rand_score(75, 10),
        "physics_score":          rand_score(82, 9),
        "chemistry_score":        rand_score(70, 11),
        "biology_score":          rand_score(55, 12),
        "english_score":          rand_score(63, 10),
        "programming_interest":   rand_skill(5.0, 1.5),
        "analytical_skills":      rand_skill(7.5, 1.2),
        "creativity_level":       rand_skill(6.5, 1.5),
        "communication_skills":   rand_skill(6.0, 1.5),
        "leadership_skills":      rand_skill(6.0, 1.5),
        "sports":                 random.choices([0, 1], weights=[0.45, 0.55])[0],
        "olympiad":               random.choices([0, 1], weights=[0.60, 0.40])[0],
        "projects_done":          random.choices(range(6), weights=[0.10, 0.15, 0.25, 0.25, 0.15, 0.10])[0],
        "course":                 "Mechanical Engineering",
    }


def profile_civil_engineering():
    return {
        "math_score":             rand_score(72, 10),
        "physics_score":          rand_score(75, 10),
        "chemistry_score":        rand_score(73, 11),
        "biology_score":          rand_score(58, 12),
        "english_score":          rand_score(65, 10),
        "programming_interest":   rand_skill(4.0, 1.5),
        "analytical_skills":      rand_skill(7.0, 1.2),
        "creativity_level":       rand_skill(6.0, 1.5),
        "communication_skills":   rand_skill(6.5, 1.5),
        "leadership_skills":      rand_skill(6.5, 1.5),
        "sports":                 random.choices([0, 1], weights=[0.50, 0.50])[0],
        "olympiad":               random.choices([0, 1], weights=[0.65, 0.35])[0],
        "projects_done":          random.choices(range(6), weights=[0.10, 0.15, 0.30, 0.25, 0.15, 0.05])[0],
        "course":                 "Civil Engineering",
    }


def profile_electrical_engineering():
    return {
        "math_score":             rand_score(78, 10),
        "physics_score":          rand_score(80, 9),
        "chemistry_score":        rand_score(68, 11),
        "biology_score":          rand_score(55, 12),
        "english_score":          rand_score(63, 10),
        "programming_interest":   rand_skill(6.5, 1.3),
        "analytical_skills":      rand_skill(8.0, 1.0),
        "creativity_level":       rand_skill(5.5, 1.5),
        "communication_skills":   rand_skill(6.0, 1.5),
        "leadership_skills":      rand_skill(6.0, 1.5),
        "sports":                 random.choices([0, 1], weights=[0.50, 0.50])[0],
        "olympiad":               random.choices([0, 1], weights=[0.50, 0.50])[0],
        "projects_done":          random.choices(range(6), weights=[0.05, 0.15, 0.25, 0.25, 0.20, 0.10])[0],
        "course":                 "Electrical Engineering",
    }


def profile_business():
    return {
        "math_score":             rand_score(65, 12),
        "physics_score":          rand_score(58, 12),
        "chemistry_score":        rand_score(58, 12),
        "biology_score":          rand_score(60, 12),
        "english_score":          rand_score(78, 9),
        "programming_interest":   rand_skill(4.5, 1.5),
        "analytical_skills":      rand_skill(7.0, 1.2),
        "creativity_level":       rand_skill(7.0, 1.5),
        "communication_skills":   rand_skill(9.0, 0.8),
        "leadership_skills":      rand_skill(8.5, 1.0),
        "sports":                 random.choices([0, 1], weights=[0.40, 0.60])[0],
        "olympiad":               random.choices([0, 1], weights=[0.70, 0.30])[0],
        "projects_done":          random.choices(range(6), weights=[0.15, 0.20, 0.25, 0.20, 0.15, 0.05])[0],
        "course":                 "Business",
    }


def profile_arts_design():
    return {
        "math_score":             rand_score(58, 12),
        "physics_score":          rand_score(55, 12),
        "chemistry_score":        rand_score(55, 12),
        "biology_score":          rand_score(60, 12),
        "english_score":          rand_score(75, 10),
        "programming_interest":   rand_skill(4.0, 1.5),
        "analytical_skills":      rand_skill(5.5, 1.5),
        "creativity_level":       rand_skill(9.5, 0.6),
        "communication_skills":   rand_skill(7.5, 1.2),
        "leadership_skills":      rand_skill(6.5, 1.5),
        "sports":                 random.choices([0, 1], weights=[0.55, 0.45])[0],
        "olympiad":               random.choices([0, 1], weights=[0.75, 0.25])[0],
        "projects_done":          random.choices(range(6), weights=[0.10, 0.20, 0.30, 0.20, 0.15, 0.05])[0],
        "course":                 "Arts & Design",
    }


# ─────────────────────────────────────────────
# Dataset assembly
# ─────────────────────────────────────────────

PROFILES = [
    profile_computer_science,
    profile_data_science,
    profile_mechanical_engineering,
    profile_civil_engineering,
    profile_electrical_engineering,
    profile_business,
    profile_arts_design,
]

# Roughly balanced classes
samples_per_class = N // len(PROFILES)
records = []

for profile_fn in PROFILES:
    for _ in range(samples_per_class):
        records.append(profile_fn())

# Fill the remainder randomly to reach exactly N rows
remainder = N - len(records)
for _ in range(remainder):
    records.append(random.choice(PROFILES)())

# Shuffle the dataset
random.shuffle(records)

df = pd.DataFrame(records)

# ─────────────────────────────────────────────
# Save to CSV
# ─────────────────────────────────────────────
OUTPUT_PATH = "dataset.csv"
df.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Dataset generated successfully!")
print(f"   Rows    : {len(df)}")
print(f"   Columns : {list(df.columns)}")
print(f"\nClass distribution:\n{df['course'].value_counts()}")
print(f"\nSaved to: {OUTPUT_PATH}")
