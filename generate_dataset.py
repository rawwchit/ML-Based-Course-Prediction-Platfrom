import pandas as pd
import random

courses = [
    "Computer Science",
    "Data Science",
    "Mechanical Engineering",
    "Civil Engineering",
    "Electrical Engineering",
    "Business",
    "Arts & Design"
]

data = []

for _ in range(500):
    math = random.randint(40, 100)
    physics = random.randint(40, 100)
    chemistry = random.randint(40, 100)
    biology = random.randint(40, 100)
    english = random.randint(40, 100)

    programming = random.randint(1, 10)
    analytical = random.randint(1, 10)
    creativity = random.randint(1, 10)
    communication = random.randint(1, 10)
    leadership = random.randint(1, 10)

    sports = random.randint(0, 1)
    olympiad = random.randint(0, 1)
    projects = random.randint(0, 5)

    # Smart logic for assigning course
    if programming > 7 and math > 75:
        course = random.choice(["Computer Science", "Data Science"])
    elif creativity > 7:
        course = "Arts & Design"
    elif math > 70 and physics > 70:
        course = random.choice(["Mechanical Engineering", "Electrical Engineering"])
    elif communication > 7 and leadership > 7:
        course = "Business"
    else:
        course = "Civil Engineering"

    data.append([
        math, physics, chemistry, biology, english,
        programming, analytical, creativity,
        communication, leadership,
        sports, olympiad, projects, course
    ])

columns = [
    "math_score", "physics_score", "chemistry_score", "biology_score", "english_score",
    "programming_interest", "analytical_skills", "creativity_level",
    "communication_skills", "leadership_skills",
    "sports", "olympiad", "projects_done", "course"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("course_dataset.csv", index=False)

print("Dataset generated successfully as course_dataset.csv")