import random
import csv
import os

# Seed for reproducibility
random.seed(42)

names = [
    "Aarav Singh", "Priya Patel", "Rohan Sharma", "Ananya Kumar", "Vikram Reddy",
    "Meera Nair", "Arjun Gupta", "Sneha Yadav", "Karan Mehta", "Divya Joshi",
    "Aditya Rao", "Kavya Iyer", "Nikhil Tiwari", "Pooja Verma", "Rahul Mishra",
    "Swati Agarwal", "Amit Srivastava", "Sunita Pandey", "Varun Chopra", "Rekha Devi",
    "Sachin Kapoor", "Geeta Bhatt", "Mohit Goyal", "Lakshmi Nair", "Rajan Pillai",
    "Anita Desai", "Prakash Rao", "Savita Yadav", "Harish Kumar", "Usha Patel",
    "Sunil Sharma", "Madhu Gupta", "Rajesh Mehta", "Kamini Singh", "Deepak Tiwari",
    "Shashi Verma", "Pankaj Agarwal", "Mamta Mishra", "Rishav Srivastava", "Chanda Devi",
    "Vinod Sahu", "Saroj Pandey", "Girish Yadav", "Kavita Nair", "Subhash Pillai",
    "Seema Bhatt", "Ashok Goyal", "Nisha Rao", "Bharat Kapoor", "Pavitra Iyer",
    "Akash Banerjee", "Ritu Ghosh", "Vivek Chatterjee", "Sunita Das", "Arnab Roy",
    "Mita Sen", "Sourav Dey", "Dipali Mukherjee", "Subrata Bose", "Rina Paul",
    "Ravi Teja", "Alekhya Raju", "Naveen Krishna", "Sravani Reddy", "Karthik Murthy",
    "Deepthi Rao", "Venkat Naidu", "Madhuri Varma", "Ganesh Hegde", "Lalitha Devi",
    "Parth Shah", "Krisha Mehta", "Dhruv Trivedi", "Nishtha Jani", "Kaushik Bhatt",
    "Mahi Pandya", "Yash Jadav", "Aarohi Chauhan", "Mitesh Prajapati", "Khushi Thakkar",
    "Tanish Khatri", "Jagriti Lal", "Devraj Nair", "Aishwarya Kulkarni", "Siddharth Jain",
    "Pallavi Dubey", "Vinayak Patil", "Chitra Naik", "Omkar Shinde", "Renuka Gaikwad",
    "Tejas Bhosale", "Manasi Deshmukh", "Tushar Jadhav", "Prachi Munde", "Abhijit Pawar",
    "Roshni Sawant", "Saurabh Wagh", "Amruta Kulkarni", "Mahesh Kadam", "Vasudha Apte"
]

genders = ["Male", "Female"]
grades = ["5", "6", "7", "8", "9", "10"]
residences = ["Rural", "Urban", "Remote"]

students = []

for i in range(100):
    student_id = f"STU{1001 + i}"
    name = names[i]
    grade = random.choice(grades)
    gender = random.choice(genders)
    age = int(grade) + random.randint(9, 12)

    attendance_pct = round(random.uniform(30, 100), 1)
    exam_score = round(random.uniform(15, 95), 1)
    distance_km = round(random.uniform(0.5, 20), 1)
    midday_meal = random.choice([0, 1])  # 1 = participates
    sibling_dropout = random.choice([0, 1])  # 1 = has sibling dropout
    family_income = random.choice(["Low", "Medium", "High"])
    residence = random.choice(residences)
    prev_year_score = round(random.uniform(15, 95), 1)
    teacher_engagement = round(random.uniform(1, 10), 1)
    extracurricular = random.choice([0, 1])

    # Heuristic risk computation (for ground-truth labels)
    risk = 0
    risk += max(0, (80 - attendance_pct) * 0.8)
    risk += max(0, (60 - exam_score) * 0.5)
    risk += min(20, distance_km * 1.0)
    risk += 10 if sibling_dropout else 0
    risk += 10 if family_income == "Low" else 0
    risk += 5 if residence == "Remote" else 0
    risk += max(0, (60 - prev_year_score) * 0.3)
    risk -= max(0, (teacher_engagement - 5) * 1.5)
    risk -= 5 if extracurricular else 0
    risk -= 5 if midday_meal else 0
    risk = max(0, min(100, round(risk, 1)))

    if risk <= 40:
        risk_label = "Low"
    elif risk <= 70:
        risk_label = "Medium"
    else:
        risk_label = "High"

    monthly_progress = []
    for m in range(6):
        monthly_progress.append(round(random.uniform(max(30, attendance_pct - 10), min(100, attendance_pct + 10)), 1))

    students.append({
        "student_id": student_id,
        "name": name,
        "grade": grade,
        "gender": gender,
        "age": age,
        "attendance_pct": attendance_pct,
        "exam_score": exam_score,
        "distance_km": distance_km,
        "midday_meal": midday_meal,
        "sibling_dropout": sibling_dropout,
        "family_income": family_income,
        "residence": residence,
        "prev_year_score": prev_year_score,
        "teacher_engagement": teacher_engagement,
        "extracurricular": extracurricular,
        "risk_score": risk,
        "risk_label": risk_label,
        "parent_contact": f"+91-98{random.randint(10000000, 99999999)}"
    })

os.makedirs("data", exist_ok=True)
output_file = "data/students.csv"

with open(output_file, "w", newline="", encoding="utf-8") as f:
    fieldnames = list(students[0].keys())
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(students)

print(f"Generated {len(students)} student records to {output_file}")
