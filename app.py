print("-" * 30)
print("---------- WELCOME-TO-AI-CREATIVE-WORLD ----------")

def normalize_skill(skill):
    return skill.strip().lower().replace(" ", "_").replace("-", "_")
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
            skill = normalize_skill(input("Enter Your Skill : "))
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
def analyze_skills(candidate):
    base_skills = ["python", "git", "linux"]
    list_of_domain = {
        "genai": base_skills + [
            "llms", "prompt_engineering", "hugging_face", "rag",
            "openai_api", "vector_databases", "embeddings",
            "langchain", "fine_tuning", "chatbots"
        ],
        "agentic_ai": base_skills + [
            "autonomous_agents", "tool_calling", "multi_agent_systems",
            "agent_memory", "agent_frameworks",
            "langgraph", "task_planning", "api_integration",
            "workflow_automation"
        ],
        "nlp": base_skills + [
            "text_preprocessing", "tokenization", "tf_idf", "ner",
            "text_similarity", "sentiment_analysis",
            "topic_modeling", "word_embeddings",
            "transformers", "bert"
        ],
        "deep_learning": base_skills + [
            "neural_networks", "backpropagation", "cnn", "rnn", "lstm",
            "transformers", "pytorch", "tensorflow",
            "model_optimization", "gpu_training"
        ],
        "computer_vision": base_skills + [
            "opencv", "image_processing", "cnn",
            "object_detection", "yolo", "image_classification",
            "face_recognition", "image_segmentation",
            "transfer_learning"
        ],
        "data_science": base_skills + [
            "pandas", "numpy", "matplotlib", "seaborn",
            "statistics", "data_cleaning",
            "exploratory_data_analysis",
            "feature_engineering", "sql", "dashboards"
        ],
        "ml_engineer": base_skills + [
            "scikit_learn", "feature_engineering",
            "model_evaluation", "pipelines",
            "mlops", "model_deployment",
            "docker", "fastapi",
            "monitoring", "cloud"
        ],
        "ai_research": base_skills + [
            "papers_reading", "math", "linear_algebra",
            "probability", "pytorch", "tensorflow",
            "ablation_studies", "optimization", "new_architectures"
        ]
    }
    domain = candidate["domain_name"]
    if domain not in list_of_domain:
        return "Invalid Domain!"
    print("You have chosen the correct trending domain!")
    resume_skills = set(normalize_skill(s) for s in candidate["skills"])
    actual_skills = set(normalize_skill(s) for s in list_of_domain[domain])
    matched_skills = resume_skills.intersection(actual_skills)
    missing_skills = actual_skills.difference(resume_skills)
    invalid_skills = resume_skills.difference(actual_skills)
    operations = {
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "invalid_skills": invalid_skills,
        "matched_count": len(matched_skills),
        "missing_count": len(missing_skills),
        "experience": candidate["experience"]
    }
    return operations
def detect_fake_profile(experience, skill_score):
    return experience >= 3 and skill_score < 50
def calculate_score(operations):
    matched = operations["matched_count"]
    missing = operations["missing_count"]
    total = matched + missing
    skill_score = round((matched / total) * 100, 2) if total else 0
    if skill_score >= 85:
        rating = "Excellent"
    elif skill_score >= 70:
        rating = "Good"
    elif skill_score >= 50:
        rating = "Average"
    else:
        rating = "Needs Improvement"
    if missing == 0:
        improvement = "Job Ready"
    elif missing <= 3:
        improvement = "Minor Upskilling Needed"
    elif missing <= 6:
        improvement = "Moderate Upskilling Needed"
    else:
        improvement = "Major Upskilling Required"
    fake_flag = detect_fake_profile(operations["experience"], skill_score)
    return {
        "skill_score": skill_score,
        "rating": rating,
        "skills_required": sorted(operations["missing_skills"]),
        "improvement_score": improvement,
        "fake_profile": fake_flag
    }
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
    if data["invalid_skills"]:
        print(f"Invalid Skills: {sorted(data['invalid_skills'])}")
    print(f"Improvement Level: {report['improvement_score']}")
    if report["fake_profile"]:
        print("⚠️ Profile Warning: Experience does not match skill level.")
    else:
        print("Profile Check: Looks genuine.")
# MAIN
candidate = get_candidate_data()
data = analyze_skills(candidate)
show_report(data)
