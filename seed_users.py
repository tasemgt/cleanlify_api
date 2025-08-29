import requests

# 🔧 Adjust to your actual server address and port
API_URL = "http://localhost:8080/register"

# 🔹 Define users to seed
users_to_seed = [
  {
    "name": "Amelia Johnson",
    "email": "ameliaj@example.com",
    "password": "password",
    "job_title": "Data Scientist",
    "location": "London",
    "organisation": "Imperial College London",
    "bio": "Specialises in healthcare data analytics and AI",
    "member_plan": "Premium Member",
    "profession": "Clinical Data Analyst"
  },
  {
    "name": "Lena Fitzgerald",
    "email": "lenafitz@example.com",
    "password": "password",
    "job_title": "NLP Scientist",
    "location": "Cambridge",
    "organisation": "Cambridge AI Lab",
    "bio": "Focus on natural language understanding and chatbots",
    "member_plan": "Premium Member",
    "profession": "Language Technology Specialist"
  },
  {
    "name": "Fatima Khan",
    "email": "fatimak@example.com",
    "password": "password",
    "job_title": "Data Analyst",
    "location": "Liverpool",
    "organisation": "NHS Digital",
    "bio": "Turning messy health records into clean insights",
    "member_plan": "Premium Member",
    "profession": "Healthcare Data Specialist"
  },
  {
    "name": "Sophie Laurent",
    "email": "sophiel@example.com",
    "password": "password",
    "job_title": "ML Ops Engineer",
    "location": "Edinburgh",
    "organisation": "Scotland AI Labs",
    "bio": "Deploying and scaling ML models",
    "member_plan": "Free Member",
    "profession": "AI Infrastructure Engineer"
  }
]

# 🔁 Send requests to /register
for user in users_to_seed:
    try:
        response = requests.post(API_URL, json=user)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Registered: {user['email']} — {result['message']}")
        else:
            print(f"❌ Failed: {user['email']} — {response.text}")
    except Exception as e:
        print(f"🔥 Error: {user['email']} — {str(e)}")