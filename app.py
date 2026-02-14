print("-" * 30)
print("---------- WELCOME-TO-AI-CREATIVE-WORLD ----------")
def get_candidate_data():
    name = input("Enter Name of the Student : ").capitalize()
    qualification = input("Enter Highest Qualification (B.tech/Degree/M.tech/M.Sc/MBA): ").strip().lower()
    year_of_pass = input("Enter Year of Passout : ")
    domain_name = input("Enter A Domain (genai/agentic_ai/nlp/deep_learning): ").strip().lower()
    experience = int(input("Enter No.of Years Experience : "))
    skills = []
    while True:
        user = input("Do You Want to add skills (yes/no): ").strip().lower()
        if user == "yes":
            skill = input("Enter Your Skill : ").strip().lower()
            skills.append(skill)
        elif user == "no":
            break
        else:
            print("Please enter only yes or no.")
    resume = {
        "name": name,
        "qualification": qualification,
        "year_of_pass": year_of_pass,
        "domain_name": domain_name,
        "experience": experience,
        "skills": skills
    }
    return resume
def analyze_skills():
    list_of_domain = {
        "genai": ["python", "llms", "prompt engineering", "hugging face", "rag"],
        "agentic_ai": ["autonomous agents", "tool calling", "multi agent systems", "agent memory", "agent frameworks"],
        "nlp": ["text preprocessing", "tokenization", "tf idf", "ner", "text similarity"],
        "deep_learning": ["neural networks", "backpropagation", "cnn", "rnn", "transformers"]
    }
    candidate = get_candidate_data()
    domain = candidate["domain_name"]
    if domain not in list_of_domain.keys():
        return "Invalid Domain! Please choose from genai, agentic_ai, nlp, deep_learning."
    print("You have chosen the correct trending domain!")
    resume_skills = set(candidate["skills"])
    actual_skills = set(list_of_domain[domain])
    matched_skills = resume_skills.intersection(actual_skills)
    missing_skills = actual_skills.difference(resume_skills)
    invalid_skills = resume_skills.difference(actual_skills)
    operations = {
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "invalid_skills": invalid_skills,
        "matched_count": len(matched_skills),
        "missing_count": len(missing_skills)
    }
    return operations
def calculate_score(operations):
    matched_count = operations["matched_count"]
    missing_count = operations["missing_count"]
    missing_skills = operations["missing_skills"]
    total_skills = matched_count + missing_count
    if total_skills == 0:
        skill_score = 0
    else:
        skill_score = round((matched_count / total_skills) * 100, 2)
    if skill_score >= 85:
        rating = "Excellent"
    elif skill_score >= 70:
        rating = "Good"
    elif skill_score >= 50:
        rating = "Average"
    else:
        rating = "Needs Improvement"
    if missing_count == 0:
        improvement = "Job Ready"
    elif missing_count <= 3:
        improvement = "Minor Upskilling Needed"
    elif missing_count <= 6:
        improvement = "Moderate Upskilling Needed"
    else:
        improvement = "Major Upskilling Required"
    report = {
        "skill_score": skill_score,
        "rating": rating,
        "skills_required": sorted(missing_skills),
        "improvement_score": improvement
    }
    return report
def show_report(data):
    if isinstance(data, str):
        print(data)
        return
    report = calculate_score(data)
    print("\n---------- RESULT REPORT ----------")
    print(f"Skill Score: {report['skill_score']}%")
    print(f"Rating: {report['rating']}")
    print(f"Matched Skills: {sorted(data['matched_skills'])}")
    print(f"Missing Skills: {sorted(data['missing_skills'])}")
    # SHOW ONLY IF INVALID SKILLS EXIST
    if data["invalid_skills"]:
        print(f"Invalid Skills (Not in Domain): {sorted(data['invalid_skills'])}")

    print(f"Improvement Level: {report['improvement_score']}")
# Main Execution
data = analyze_skills()
show_report(data)