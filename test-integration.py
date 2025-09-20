import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration - can test locally or deployed version
AI_AGENT_URL = os.getenv('AI_AGENT_URL', 'http://localhost:8080')
MOCK_API_URL = os.getenv('MOCK_API_URL', 'http://localhost:8081')

def test_mock_api_health():
    """Test if mock data API is healthy"""
    try:
        response = requests.get(f"{MOCK_API_URL}/health")
        if response.status_code == 200:
            print("✅ Mock Data API is healthy")
            return True
        else:
            print(f"❌ Mock Data API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Mock Data API: {e}")
        return False

def test_ai_agent_health():
    """Test if AI agent is healthy"""
    try:
        response = requests.get(f"{AI_AGENT_URL}/health")
        if response.status_code == 200:
            print("✅ AI Agent is healthy")
            return True
        else:
            print(f"❌ AI Agent health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to AI Agent: {e}")
        return False

def test_career_analysis():
    """Test the complete flow from mock data to AI analysis"""
    
    test_data = {
        "userId": "user123",
        "mockDataApiUrl": MOCK_API_URL
    }
    
    print("\n🧪 Testing CareerCompass AI Analysis...")
    print(f"📊 Testing with userId: {test_data['userId']}")
    
    try:
        response = requests.post(f"{AI_AGENT_URL}/api/analyze-career", json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ AI Analysis Successful!")
            print(f"\n📨 Analysis Results:")
            print(f"  - User Profile: {result.get('userProfile', 'N/A')}")
            print(f"  - Primary Goal: {result['analysis']['primaryGoal']}")
            print(f"  - Recommended Skills: {', '.join(result['analysis']['recommendedSkills'])}")
            print(f"  - Financial Advice: {result['analysis']['financialAdvice']}")
            print(f"\n📚 Suggested Courses:")
            for course in result['analysis']['suggestedCourses']:
                print(f"  - {course['name']} ({course['provider']}) - {course['estimatedCost']}")
            print(f"\n🎯 Next Steps:")
            for step in result['analysis']['nextSteps']:
                print(f"  - {step}")
            return True
        else:
            print(f"❌ Request failed with status: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def test_user_list():
    """Test fetching available user profiles"""
    try:
        response = requests.get(f"{MOCK_API_URL}/api/users")
        if response.status_code == 200:
            users = response.json()
            print("\n👥 Available test users:")
            for user in users:
                print(f"  - {user['userId']}: {user['name']} ({user['profile']})")
            return True
        else:
            print(f"❌ Failed to fetch users: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot fetch users: {e}")
        return False

if __name__ == "__main__":
    print("🚀 CareerCompass Integration Test Suite")
    print("=" * 40)
    
    # Run all tests
    all_passed = True
    
    # Test 1: Mock API Health
    if not test_mock_api_health():
        all_passed = False
    
    # Test 2: AI Agent Health
    if not test_ai_agent_health():
        all_passed = False
    
    # Test 3: User List
    if all_passed:
        test_user_list()
    
    # Test 4: Career Analysis
    if all_passed:
        success = test_career_analysis()
        if not success:
            all_passed = False
    
    # Final result
    if all_passed:
        print("\n🎉 All tests passed! Your CareerCompass backend is ready!")
    else:
        print("\n🔧 Some tests failed. Please check your services and try again.")
