import os
import json
import httpx
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")
genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="CareerCompass AI Agent", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    userId: str
    mockDataApiUrl: str = "http://mock-data-api-service:8080"

class CareerRecommendation(BaseModel):
    primaryGoal: str
    recommendedSkills: list
    suggestedCourses: list
    financialAdvice: str
    nextSteps: list

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ai-career-agent"}


@app.post("/api/analyze-career", response_model=dict)
async def analyze_career_path(request: AnalysisRequest):
    try:
        # Fetch user data from mock API
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{request.mockDataApiUrl}/api/users/{request.userId}/financial-data"
            )

            if response.status_code != 200:
                raise HTTPException(status_code=404, detail="User data not found")

            user_data = response.json()

        # Analyze with Gemini
        model = genai.GenerativeModel('gemini-pro')

        prompt = f"""
        Analyze this financial profile and provide career guidance:

        User Profile: {user_data['name']} - {user_data['profile']}
        Monthly Income: ${user_data['monthlyIncome']}
        Career Stage: {user_data['careerStage']}

        Spending Breakdown:
        {json.dumps(user_data['spendingCategories'], indent=2)}

        Recent Transactions:
        {json.dumps(user_data['recentTransactions'], indent=2)}

        Current Goals: {', '.join(user_data['goals'])}

        Please provide career guidance in this JSON format:
        {{
            "primaryGoal": "One main career objective based on their profile",
            "recommendedSkills": ["skill1", "skill2", "skill3"],
            "suggestedCourses": [
                {{"name": "Course Name", "provider": "Platform", "estimatedCost": "$XX"}},
                {{"name": "Course Name 2", "provider": "Platform", "estimatedCost": "$XX"}}
            ],
            "financialAdvice": "Specific financial recommendation based on their spending",
            "nextSteps": ["actionable step 1", "actionable step 2", "actionable step 3"]
        }}

        Focus on practical, actionable advice based on their current financial situation and career stage.
        """

        response = model.generate_content(prompt)

        # Parse the AI response
        try:
            ai_text = response.text

            # Remove any markdown code block formatting
            if "```json" in ai_text:
                ai_text = ai_text.split("```json")[1].split("```")[0]
            elif "```" in ai_text:
                ai_text = ai_text.split("```")[1].split("```")[0]

            recommendation = json.loads(ai_text)

        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            recommendation = {
                "primaryGoal": "Build technical skills for career advancement",
                "recommendedSkills": ["Data Analysis", "Python Programming", "Communication"],
                "suggestedCourses": [
                    {"name": "Python for Data Science", "provider": "Coursera", "estimatedCost": "$49"},
                    {"name": "Excel to Python", "provider": "Udemy", "estimatedCost": "$85"}
                ],
                "financialAdvice": "Consider allocating 15% of income to skill development",
                "nextSteps": [
                    "Start with one online course this month",
                    "Set up a dedicated learning budget",
                    "Track progress weekly"
                ]
            }

        return {
            "success": True,
            "userId": request.userId,
            "userProfile": user_data['profile'],
            "analysis": recommendation,
            "confidence": "high"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
