print("-" * 30)
print("---------- WELCOME-TO-AI-CREATIVE-WORLD ----------")

import webbrowser

def normalize_skill(skill):
    return skill.strip().lower().replace(" ", "_").replace("-", "_")
domain_recommendations = {
    "genai": {
        "python": "https://www.youtube.com/watch?v=rfscVS0vtbw",
        "git": "https://www.youtube.com/watch?v=RGOj5yH7evk",
        "linux": "https://www.youtube.com/watch?v=wBp0Rb-ZJak",
        "llms": "https://www.youtube.com/watch?v=zjkBMFhNj_g",
        "prompt engineering": "https://www.youtube.com/watch?v=_ZvnD73m40o",
        "hugging face": "https://www.youtube.com/watch?v=QdY3A_c6g4U",
        "rag": "https://www.youtube.com/watch?v=T-D1OfcDW1M",
        "openai api": "https://www.youtube.com/watch?v=ZBKpAp_6TGI",
        "vector databases": "https://www.youtube.com/watch?v=dN0lsF2cvm4",
        "embeddings": "https://www.youtube.com/watch?v=Z_Bk6mE3sJE",
        "langchain": "https://www.youtube.com/watch?v=aywZrzNaKjs",
        "fine tuning": "https://www.youtube.com/watch?v=eC6Hd1hFvos",
        "chatbots": "https://www.youtube.com/watch?v=QdY3A_c6g4U"
    },
    "agentic_ai": {
        "python": "https://www.youtube.com/watch?v=rfscVS0vtbw",
        "git": "https://www.youtube.com/watch?v=RGOj5yH7evk",
        "linux": "https://www.youtube.com/watch?v=wBp0Rb-ZJak",
        "autonomous agents": "https://www.youtube.com/watch?v=example1",
        "tool calling": "https://www.youtube.com/watch?v=V3l3K8v3HnE",
        "multi agent systems": "https://www.youtube.com/watch?v=7V2gYc3Z4mQ",
        "agent memory": "https://www.youtube.com/watch?v=8kFhFJbF5kA",
        "agent frameworks": "https://www.youtube.com/watch?v=example2",
        "langgraph": "https://www.youtube.com/watch?v=example3",
        "task planning": "https://www.youtube.com/watch?v=1i7pI6p0r3k",
        "api integration": "https://www.youtube.com/watch?v=example4",
        "workflow automation": "https://www.youtube.com/watch?v=9q0z8rF9pZg"
    },
    "nlp": {
        "python": "https://www.youtube.com/watch?v=rfscVS0vtbw",
        "git": "https://www.youtube.com/watch?v=RGOj5yH7evk",
        "linux": "https://www.youtube.com/watch?v=wBp0Rb-ZJak",
        "text preprocessing": "https://www.youtube.com/watch?v=example5",
        "tokenization": "https://www.youtube.com/watch?v=9kXk3rFf3kA",
        "tf idf": "https://www.youtube.com/watch?v=4vGyIo7lD1k",
        "ner": "https://www.youtube.com/watch?v=example6",
        "text similarity": "https://www.youtube.com/watch?v=example7",
        "sentiment analysis": "https://www.youtube.com/watch?v=ujId4ipkBio",
        "topic modeling": "https://www.youtube.com/watch?v=example8",
        "word embeddings": "https://www.youtube.com/watch?v=viZrOnJclY0",
        "transformers": "https://www.youtube.com/watch?v=SZorAJ4I-sA",
        "bert": "https://www.youtube.com/watch?v=xI0HHN5XKDo"
    },
    "deep_learning": {
        "python": "https://www.youtube.com/watch?v=rfscVS0vtbw",
        "git": "https://www.youtube.com/watch?v=RGOj5yH7evk",
        "linux": "https://www.youtube.com/watch?v=wBp0Rb-ZJak",
        "neural networks": "https://www.youtube.com/watch?v=aircAruvnKk",
        "backpropagation": "https://www.youtube.com/watch?v=Ilg3gGewQ5U",
        "cnn": "https://www.youtube.com/watch?v=YRhxdVk_sIs",
        "rnn": "https://www.youtube.com/watch?v=WCUNPb-5EYI",
        "lstm": "https://www.youtube.com/watch?v=YCzL96nL7j0",
        "transformers": "https://www.youtube.com/watch?v=SZorAJ4I-sA",
        "pytorch": "https://www.youtube.com/watch?v=V_xro1bcAuA",
        "tensorflow": "https://www.youtube.com/watch?v=tPYj3fFJGjk",
        "model optimization": "https://www.youtube.com/watch?v=IHZwWFHWa-w",
        "gpu training": "https://www.youtube.com/watch?v=example9"
    },
    "computer_vision": {
        "python": "https://www.youtube.com/watch?v=rfscVS0vtbw",
        "git": "https://www.youtube.com/watch?v=RGOj5yH7evk",
        "linux": "https://www.youtube.com/watch?v=wBp0Rb-ZJak",
        "opencv": "https://www.youtube.com/watch?v=oXlwWbU8l2o",
        "image processing": "https://www.youtube.com/watch?v=9-3eG9xYgU0",
        "cnn": "https://www.youtube.com/watch?v=YRhxdVk_sIs",
        "object detection": "https://www.youtube.com/watch?v=MPU2HistivI",
        "yolo": "https://www.youtube.com/watch?v=8hL0Gk0tKfI",
        "image segmentation": "https://www.youtube.com/watch?v=IHZwWFHWa-w",
        "transfer learning": "https://www.youtube.com/watch?v=yofjFQddwHE",
        "face recognition": "https://www.youtube.com/watch?v=88HdqNDQsEk",
        "image classification": "https://www.youtube.com/watch?v=example10",
        "data augmentation": "https://www.youtube.com/watch?v=ZcL9K0X5YJk"
    },
    "data_science": {
        "python": "https://www.youtube.com/watch?v=rfscVS0vtbw",
        "git": "https://www.youtube.com/watch?v=RGOj5yH7evk",
        "linux": "https://www.youtube.com/watch?v=wBp0Rb-ZJak",
        "numpy": "https://www.youtube.com/watch?v=QUT1VHiLmmI",
        "pandas": "https://www.youtube.com/watch?v=vmEHCJofslg",
        "matplotlib": "https://www.youtube.com/watch?v=DAQNHzOcO5A",
        "seaborn": "https://www.youtube.com/watch?v=6GUZXDef2U0",
        "statistics": "https://www.youtube.com/watch?v=xxpc-HPKN28",
        "data cleaning": "https://www.youtube.com/watch?v=RjVvV7M9q6w",
        "exploratory data analysis": "https://www.youtube.com/watch?v=xi0vhXFPegw",
        "feature engineering": "https://www.youtube.com/watch?v=7eB7X3cPpD8",
        "sql": "https://www.youtube.com/watch?v=HXV3zeQKqGY",
        "dashboards": "https://www.youtube.com/watch?v=example11",
        "eda": "https://www.youtube.com/watch?v=xi0vhXFPegw",
        "machine learning": "https://www.youtube.com/watch?v=7eh4d6sabA0"
    },
    "ml_engineer": {
        "python": "https://www.youtube.com/watch?v=rfscVS0vtbw",
        "git": "https://www.youtube.com/watch?v=RGOj5yH7evk",
        "linux": "https://www.youtube.com/watch?v=wBp0Rb-ZJak",
        "scikit learn": "https://www.youtube.com/watch?v=0Lt9w-BxKFQ",
        "feature engineering": "https://www.youtube.com/watch?v=7eB7X3cPpD8",
        "model evaluation": "https://www.youtube.com/watch?v=example12",
        "pipelines": "https://www.youtube.com/watch?v=example13",
        "mlops": "https://www.youtube.com/watch?v=06-AZXmwHjo",
        "model deployment": "https://www.youtube.com/watch?v=UbCWoMf80PY",
        "docker": "https://www.youtube.com/watch?v=fqMOX6JJhGo",
        "fastapi": "https://www.youtube.com/watch?v=example14",
        "monitoring": "https://www.youtube.com/watch?v=example15",
        "cloud": "https://www.youtube.com/watch?v=ulprqHHWlng"
    },
    "ai_research": {
        "python": "https://www.youtube.com/watch?v=rfscVS0vtbw",
        "git": "https://www.youtube.com/watch?v=RGOj5yH7evk",
        "linux": "https://www.youtube.com/watch?v=wBp0Rb-ZJak",
        "papers reading": "https://www.youtube.com/watch?v=733m6qBH-jI",
        "math": "https://www.youtube.com/watch?v=LN7cCW1rSsI",
        "linear algebra": "https://www.youtube.com/watch?v=kjBOesZCoqc",
        "probability": "https://www.youtube.com/watch?v=SkidyDQuupA",
        "pytorch": "https://www.youtube.com/watch?v=V_xro1bcAuA",
        "tensorflow": "https://www.youtube.com/watch?v=tPYj3fFJGjk",
        "ablation studies": "https://www.youtube.com/watch?v=example16",
        "optimization": "https://www.youtube.com/watch?v=example17",
        "new architectures": "https://www.youtube.com/watch?v=example18"
    }
}

# Normalize all keys
for domain, skills_dict in domain_recommendations.items():
    domain_recommendations[domain] = {normalize_skill(skill): url for skill, url in skills_dict.items()}

# Normalize all keys in domain_recommendations to match normalized skill names
for domain in domain_recommendations:
    domain_recommendations[domain] = {
        normalize_skill(skill): url
        for skill, url in domain_recommendations[domain].items()
    }
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
    quiz_percent = round((quiz_score / 5) * 100, 2)

    grade, tag = get_grade_and_tag(quiz_percent)

    print("\n---------- QUIZ RESULT ----------")
    print(f"Quiz Score: {quiz_score}/5")
    print(f"Quiz Percentage: {quiz_percent}%")
    print(f"Grade: {grade}")
    print(f"Tag: {tag}")

    print("\n---------- SKILL GAP REPORT ----------")
    report = calculate_score(data, quiz_percent)

    print(f"Resume Skill Score: {report['skill_score']}%")
    print(f"Rating: {report['rating']}")
    print(f"Matched Skills: {sorted(data['matched_skills'])}")
    print(f"Missing Skills: {report['skills_required']}")
    print(f"Improvement Level: {report['improvement_score']}")

    if report["fake_profile"]:
        print("⚠️ Profile Warning: Experience does not match skill level.")
    else:
        print("Profile looks genuine")

    # 🔥 Recommended Learning Resources Section
    selected_domain = candidate["domain_name"]
    missing_skills = report["skills_required"]

    if selected_domain in domain_recommendations:
        domain_links = domain_recommendations[selected_domain]

        print("\n📚 Recommended Learning Resources:")
        for skill in missing_skills:
            skill_key = normalize_skill(skill)
            url = domain_links.get(skill_key)
            if url:
                print(f"- {skill} → {url}")
            else:
                print(f"- {skill} → No direct resource available")

    # Final combined judgment
    if quiz_percent >= 70 and report["skill_score"] >= 70:
        print("\nFinal Review: Strong Candidate 🔥")
    elif quiz_percent >= 50 or report["skill_score"] >= 50:
        print("\nFinal Verdict: Potential Candidate ⚡")
    else:
        print("\nFinal Review: Needs Serious Upskilling 📚")


# EXECUTION
candidate = get_candidate_data()
data = analyze_skills(candidate)

if isinstance(data, str):
    print(data)
else:
    show_report(data, candidate)

