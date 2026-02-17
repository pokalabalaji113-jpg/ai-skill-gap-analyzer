print("-" * 30)
print("---------- WELCOME-TO-AI-CREATIVE-WORLD ----------")

def normalize_skill(skill):
    return skill.strip().lower().replace(" ", "_").replace("-", "_")
def get_candidate_data():
    name = input("Enter Name of the Student : ").capitalize()
    qualification = input("Enter Highest Qualification (Btech/Degree/Mtech/MSc/MBA): ").strip().lower().replace(" ","").replace("-","").replace("_","")
    from datetime import datetime
    current_year = datetime.now().year
    while True:
        try:
            year_of_pass = int(input("Enter Year of Passout : "))
            break
        except ValueError:
            print("Please enter a valid year.")
    if year_of_pass > current_year + 2 or year_of_pass < current_year - 20:
        print("Not Eligible: Year of passout is invalid.")
        exit()
    domain_name = input("Enter A Domain (genai/agentic_ai/nlp/deep_learning/computer_vision/data_science/ml_engineer/ai_research): ").strip().lower().replace(" ", "_").replace("-", "_")   
    skills = []
    while True:
        try:
            experience = int(input("Enter No.of Years Experience : "))
            break
        except ValueError:
            print("Please enter a valid number.")
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
    #Basic skills Everyone to Learn
    base_skills = ["python", "git","linux"]
    list_of_domain = {
        "genai": base_skills + [
            "llms", "prompt_engineering", "hugging_face", 
            "rag","openai_api", "vector_databases", "embeddings",
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
def quiz_percentage(candidate):
    domain_questions = {
        "genai": [
            "GenAI works mainly using which language?","LLM stands for?","What is RAG used for?",
            "Which library is used for GenAI pipelines?","What is an embedding?"
        ],
        "agentic_ai": [
            "What is an autonomous agent?","What is tool calling?","What is multi-agent system?",
            "What is agent memory?","What is task planning?"
        ],
        "nlp": [
            "What is tokenization?","What does NER mean?","What is TF-IDF?",
            "What is sentiment analysis?","What is BERT?"
        ],
        "deep_learning": [
            "What is a neural network?","What is backpropagation?","What is CNN used for?",
            "What is RNN used for?","What is GPU training?"
        ],
        "computer_vision": [
            "What is OpenCV?","What is object detection?","What is YOLO?",
            "What is image segmentation?","What is face recognition?"
        ],
        "data_science": [
            "What is Pandas used for?","What is NumPy?","What is data cleaning?",
            "What is EDA?","What is SQL?"
        ],
        "ml_engineer": [
            "What is Scikit-learn?","What is model deployment?","What is Docker?",
            "What is MLOps?","What is monitoring?"
        ],
        "ai_research": [
            "What is linear algebra?","What is probability?","What is PyTorch?",
            "What is an ablation study?","What is optimization?"
        ]
    }

    options = {
        "genai": [
            "A) Python\nB) HTML\nC) CSS",
            "A) Low Level Machine\nB) Large Language Model\nC) Long Logic Method",
            "A) Random Access Graph\nB) Retrieval Augmented Generation\nC) Real AI Generator",
            "A) NumPy\nB) Django\nC) LangChain",
            "A) Image\nB) Database\nC) Vector representation"
        ],
        "agentic_ai": [
            "A) Website\nB) Self working system\nC) Database",
            "A) Calling people\nB) API error\nC) Using tools by agent",
            "A) Multiple users\nB) Multiple models\nC) Multiple agents working",
            "A) Training data\nB) Logs\nC) Storing past info",
            "A) Debugging\nB) Planning steps\nC) Testing"
        ],
        "nlp": [
            "A) Training model\nB) Translation\nC) Splitting text",
            "A) Neural Engine\nB) Named Entity Recognition\nC) Network Error",
            "A) Sorting\nB) Clustering\nC) Word weighting",
            "A) Token split\nB) Emotion detection\nC) Indexing",
            "A) Database\nB) API\nC) NLP model"
        ],
        "deep_learning": [
            "A) Database\nB) Server\nC) AI model",
            "A) Deployment\nB) Error correction\nC) Testing",
            "A) Text tasks\nB) Audio\nC) Image tasks",
            "A) Images\nB) Tables\nC) Sequential data",
            "A) Storage\nB) Backup\nC) Faster training"
        ],
        "computer_vision": [
            "A) OS\nB) Database\nC) Vision library",
            "A) Storing images\nB) Cleaning data\nC) Finding objects",
            "A) Language model\nB) API\nC) Detection model",
            "A) Compressing\nB) Resizing\nC) Dividing image",
            "A) Storing faces\nB) Drawing\nC) Identifying faces"
        ],
        "data_science": [
            "A) Web dev\nB) Gaming\nC) Data handling",
            "A) UI library\nB) API\nC) Math library",
            "A) Training\nB) Testing\nC) Fixing data",
            "A) Deleting\nB) Uploading\nC) Exploring data",
            "A) OS\nB) Browser\nC) Query language"
        ],
        "ml_engineer": [
           "A) Game engine\nB) OS\nC) ML library",
            "A) Training\nB) Testing\nC) Putting model live",
            "A) Database\nB) Browser\nC) Container tool",
            "A) ML output\nB) ML object\nC) ML operations",
            "A) Storing files\nB) Logging\nC) Tracking models"
        ],
        "ai_research": [
            "A) Database\nB) API\nC) Math for vectors",
            "A) Sorting\nB) Storage\nC) Chance math",
            "A) Browser\nB) OS\nC) DL framework",
            "A) Training\nB) Testing\nC) Removing components",
            "A) Debugging\nB) Logging\nC) Improving performance"
        ]
    }

    answers = {
    "genai": ["A","B","B","C","C"],
        "agentic_ai": ["B","C","C","C","B"],
        "nlp": ["C","B","C","B","C"],
        "deep_learning": ["C","B","C","C","C"],
        "computer_vision": ["C","C","C","C","C"],
        "data_science": ["C","C","C","C","C"],
        "ml_engineer": ["C","C","C","C","C"],
        "ai_research": ["C","C","C","C","C"]   
    }
    user_answers=[]
    score=0
    domain=candidate["domain_name"]
    if domain not in domain_questions:
       print("Invalid domain for quiz.")
       return 0
#if domain in domain_questions.keys():
    for i in range (len(domain_questions[domain])):
        print(domain_questions[domain][i])
        print(options[domain][i])
        user_choice=input("Choose Any Option (A/B/C) : ").strip().upper()
        user_answers.append(user_choice)  
    for i in range(len(answers[domain])):
        if user_answers[i] == answers[domain][i]:
            score+=1
    return score

def detect_fake_profile(experience, skill_score,quiz_percentage):
    return experience >= 3 and (skill_score < 50 or quiz_percentage < 50)

def calculate_percentage(score, total):
    return (score / total) * 100
def get_grade_and_tag(percentage):
    if percentage >= 90:
        return "A+", "Pro Coder"
    elif percentage >= 80:
        return "A", "Advanced"
    elif percentage >= 70:
        return "B+", "Intermediate+"
    elif percentage >= 60:
        return "B", "Intermediate"
    elif percentage >= 50:
        return "C", "Beginner"
    else:
        return "D", "Needs Practice"

def calculate_score(operations,quiz_percent):
    matched = operations["matched_count"]
    missing = operations["missing_count"]
    total = matched + missing
    skill_score = round((matched / total) * 100, 2) if total>0 else 0
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
    fake_flag = detect_fake_profile(operations["experience"], skill_score,quiz_percent)
    return {
        "skill_score": skill_score,
        "rating": rating,
        "skills_required": sorted(operations["missing_skills"]),
        "improvement_score": improvement,
        "fake_profile": fake_flag
    }

def show_report(data, candidate):
    print("\n---------- QUIZ ROUND ----------")
    quiz_score = quiz_percentage(candidate)
    quiz_percent = (quiz_score / 5) * 100

    grade, tag = get_grade_and_tag(quiz_percent)

    print("\n---------- QUIZ RESULT ----------")
    print("Quiz Score:", quiz_score, "/5")
    print("Quiz Percentage:", quiz_percent)
    print("Grade:", grade)
    print("Tag:", tag)

    print("\n---------- SKILL GAP REPORT ----------")
    report = calculate_score(data,quiz_percent)

    print("Resume Skill Score:", report["skill_score"], "%")
    print("Rating:", report["rating"])
    print("Matched Skills:", sorted(data["matched_skills"]))
    print("Missing Skills:", report["skills_required"])
    print("Improvement Level:", report["improvement_score"])

    if report["fake_profile"]:
        print("⚠️ Profile Warning: Experience does not match skill level.")
    else:
        print("Profile looks genuine")

    # Final combined judgment (this is GOLD for interviews)
    if quiz_percent >= 70 and report["skill_score"] >= 70:
        print("Final Review: Strong Candidate 🔥")
    elif quiz_percent >= 50:
        print("Final Verdict: Potential Candidate ⚡")
    else:
        print("Final Review: Needs Serious Upskilling 📚")

# MAIN
# MAIN EXECUTION
candidate = get_candidate_data()
data = analyze_skills(candidate)

if isinstance(data, str):
    print(data)
else:
    show_report(data, candidate)

