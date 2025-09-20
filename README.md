# CareerCompass Backend Services

A microservices-based career guidance platform that analyzes financial data and provides personalized career recommendations using Google's Gemini AI.

## Project Structure

```
careercompass-services/
├── ai-agent/               # AI-powered career analysis service
│   ├── main.py            # FastAPI application
│   ├── requirements.txt   # Python dependencies
│   ├── Dockerfile         # Container configuration
│   └── .env              # Environment variables
├── mock-data-api/         # Mock financial data service
│   ├── index.js          # Express application
│   ├── package.json      # Node.js dependencies
│   ├── Dockerfile        # Container configuration
│   └── .env             # Environment variables (if needed)
├── careercompass-deployment.yaml  # Kubernetes deployment
└── test-integration.py           # Integration test suite
```

## Services Overview

### 1. Mock Data API Service
- **Port**: 8081 (local), 8080 (container)
- **Technology**: Node.js + Express
- **Purpose**: Provides mock financial data for testing
- **Endpoints**:
  - `GET /health` - Health check
  - `GET /api/users` - List all test users
  - `GET /api/users/:userId/financial-data` - Get user financial data

### 2. AI Agent Service
- **Port**: 8080
- **Technology**: Python + FastAPI
- **Purpose**: Analyzes financial data and provides career recommendations
- **Endpoints**:
  - `GET /health` - Health check
  - `POST /api/analyze-career` - Analyze career path

## Setup Instructions

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker (optional, for containerization)
- Google Gemini API key

### Local Development Setup

#### 1. Clone and navigate to the project
```bash
cd careercompass-services
```

#### 2. Set up the Mock Data API
```bash
cd mock-data-api
npm install
node index.js
```
The service will run on http://localhost:8080

#### 3. Set up the AI Agent (in a new terminal)
```bash
cd ai-agent
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
python main.py
```
The service will run on http://localhost:8080

**Note**: To run both services locally, you'll need to change one of the ports. Update the mock-data-api to run on port 8081:

In `mock-data-api/index.js`, change:
```javascript
const port = 8081;  // Changed from 8080
```

#### 4. Environment Variables
Make sure your `ai-agent/.env` file contains:
```
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

### Running Integration Tests

With both services running:
```bash
python test-integration.py
```

For testing with custom URLs:
```bash
# Create .env file in root directory
echo "AI_AGENT_URL=http://localhost:8080" > .env
echo "MOCK_API_URL=http://localhost:8081" >> .env
python test-integration.py
```

## API Documentation

### Mock Data API

#### Get User Financial Data
```http
GET /api/users/:userId/financial-data
```

Response:
```json
{
  "userId": "user123",
  "name": "Alex Johnson",
  "profile": "Recent Graduate",
  "monthlyIncome": 3200,
  "spendingCategories": {
    "education": 450,
    "food": 400,
    "rent": 1200,
    "entertainment": 200,
    "transportation": 150,
    "savings": 800
  },
  "recentTransactions": [...],
  "careerStage": "entry-level",
  "goals": ["learn new skills", "increase income", "build emergency fund"]
}
```

### AI Agent API

#### Analyze Career Path
```http
POST /api/analyze-career
Content-Type: application/json

{
  "userId": "user123",
  "mockDataApiUrl": "http://localhost:8081"
}
```

Response:
```json
{
  "success": true,
  "userId": "user123",
  "userProfile": "Recent Graduate",
  "analysis": {
    "primaryGoal": "Build technical skills for career advancement",
    "recommendedSkills": ["Data Analysis", "Python Programming", "Communication"],
    "suggestedCourses": [
      {
        "name": "Python for Data Science",
        "provider": "Coursera",
        "estimatedCost": "$49"
      }
    ],
    "financialAdvice": "Consider allocating 15% of income to skill development",
    "nextSteps": [
      "Start with one online course this month",
      "Set up a dedicated learning budget",
      "Track progress weekly"
    ]
  },
  "confidence": "high"
}
```

## Docker Deployment

### Build Images
```bash
# Build Mock Data API
cd mock-data-api
docker build -t mock-data-api:latest .

# Build AI Agent
cd ../ai-agent
docker build -t ai-agent:latest .
```

### Run with Docker Compose

Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  mock-data-api:
    build: ./mock-data-api
    ports:
      - "8081:8080"
    
  ai-agent:
    build: ./ai-agent
    ports:
      - "8080:8080"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    depends_on:
      - mock-data-api
```

Run:
```bash
docker-compose up
```

## Kubernetes Deployment

The `careercompass-deployment.yaml` file contains the Kubernetes configuration. Before deploying:

1. Build and push images to your container registry
2. Update the image URLs in the deployment file
3. Create a secret for the Gemini API key:
```bash
kubectl create secret generic gemini-secret --from-literal=api-key=your_actual_key_here
```

4. Update the deployment to use the secret:
```yaml
env:
- name: GEMINI_API_KEY
  valueFrom:
    secretKeyRef:
      name: gemini-secret
      key: api-key
```

5. Deploy:
```bash
kubectl apply -f careercompass-deployment.yaml
```

## Security Considerations

1. **API Keys**: Never commit API keys to version control. Use environment variables or secrets management.
2. **CORS**: Currently allows all origins (`*`). Restrict this in production.
3. **Input Validation**: Add proper validation for all API inputs.
4. **Rate Limiting**: Implement rate limiting to prevent abuse.
5. **HTTPS**: Use HTTPS in production environments.

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   - Change the port in the respective service configuration
   - Kill the process using the port: `netstat -ano | findstr :8080` (Windows)

2. **Gemini API Key Not Working**
   - Verify the key is correct
   - Check if the key has proper permissions
   - Ensure the .env file is in the correct format (no quotes needed)

3. **Services Can't Communicate**
   - Check if both services are running
   - Verify the URLs in the configuration
   - Check firewall settings

## Next Steps for Frontend Development

Your backend is now ready! Here's what the frontend needs to integrate:

1. **CORS is enabled** - Frontend can make requests from any origin (update this for production)
2. **API Endpoints** - Use the documented endpoints above
3. **Test Users** - Use "user123" or "user456" for testing
4. **Error Handling** - Backend returns proper HTTP status codes and error messages

## License

This project is a prototype for educational purposes.