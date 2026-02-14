print("-" * 30)
print("---------- WELCOME-TO-AI-CREATIVE-WORLD ----------")
def get_candidate_data():
    name = input("Enter Name of the Student : ").capitalize()
    qualification = input("Enter Highest Qualification (B.tech/Degree/M.tech/M.Sc/MBA): ").strip().lower()
    year_of_pass = input("Enter Year of Passout : ")
    domain_name = input("Enter A Domain (genai/agentic_ai/nlp/deep_learning/computer_vision/data_science/ml_engineer/ai_research): ").strip().lower()
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
    base_skills = ["python", "git", "linux"]
    list_of_domain = {

    "genai": base_skills+[
        "llms", "prompt engineering", "hugging face", "rag",
        "openai api", "vector databases", "embeddings",
        "langchain", "fine tuning", "chatbots"
    ],

    "agentic_ai": base_skills+[
        "autonomous agents", "tool calling", "multi agent systems",
        "agent memory", "agent frameworks",
        "langgraph", "task planning", "api integration",
        "workflow automation"
    ],

    "nlp": base_skills+[
        "text preprocessing", "tokenization", "tf idf", "ner",
        "text similarity", "sentiment analysis",
        "topic modeling", "word embeddings",
        "transformers", "bert"
    ],

    "deep_learning": base_skills+[
        "neural networks", "backpropagation", "cnn", "rnn", "lstm",
        "transformers", "pytorch", "tensorflow",
        "model optimization", "gpu training"
    ],

    "computer_vision": base_skills+[
        "opencv", "image processing", "cnn",
        "object detection", "yolo", "image classification",
        "face recognition", "image segmentation",
        "transfer learning"
    ],

    "data_science": base_skills+[
        "python", "pandas", "numpy", "matplotlib", "seaborn",
        "statistics", "data cleaning", "exploratory data analysis",
        "feature engineering", "sql", "dashboards"
    ],

    "ml_engineer": base_skills+[
        "scikit-learn", "feature engineering",
        "model evaluation", "pipelines",
        "mlops", "model deployment",
        "docker", "fastapi",
        "monitoring", "cloud"
    ],

    "ai_research": base_skills+[
        "papers reading", "math", "linear algebra",
        "probability", "pytorch", "tensorflow",
        "experiments", "ablation studies",
        "optimization", "new architectures"
    ]
}

    candidate = get_candidate_data()
    domain = candidate["domain_name"]
    if domain not in list_of_domain.keys():
        return "Invalid Domain! Please choose from genai, agentic_ai, nlp, deep_learning,computer_vision,data_science,ml_engineer,ai_research"
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