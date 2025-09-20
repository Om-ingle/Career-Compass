   const express = require('express');
   const cors = require('cors');
   const app = express();
   const port = process.env.PORT || 8081;  // Changed to 8081 for local development

   // Enable CORS for all routes
   app.use(cors());
   app.use(express.json());

   // Enhanced mock data with multiple user profiles
   const mockUsers = {
     "user123": {
       userId: "user123",
       name: "Alex Johnson",
       profile: "Recent Graduate",
       monthlyIncome: 3200,
       spendingCategories: {
         "education": 450,
         "food": 400,
         "rent": 1200,
         "entertainment": 200,
         "transportation": 150,
         "savings": 800
       },
       recentTransactions: [
         { id: 1, amount: -89, description: "Coursera Subscription", category: "education" },
         { id: 2, amount: -45, description: "Programming Book", category: "education" },
         { id: 3, amount: 3200, description: "Salary Deposit", category: "income" }
       ],
       careerStage: "entry-level",
       goals: ["learn new skills", "increase income", "build emergency fund"]
     },
     "user456": {
       userId: "user456",
       name: "Sarah Chen",
       profile: "Career Changer",
       monthlyIncome: 4500,
       spendingCategories: {
         "education": 800,
         "food": 500,
         "rent": 1500,
         "entertainment": 150,
         "transportation": 200,
         "savings": 1350
       },
       recentTransactions: [
         { id: 1, amount: -299, description: "Data Science Bootcamp", category: "education" },
         { id: 2, amount: -150, description: "Technical Books", category: "education" },
         { id: 3, amount: 4500, description: "Salary Deposit", category: "income" }
       ],
       careerStage: "transitioning",
       goals: ["switch careers", "learn data science", "increase technical skills"]
     }
   };

   // Health check endpoint
   app.get('/health', (req, res) => {
     res.json({ status: 'healthy', service: 'mock-data-api' });
   });

   // Get user financial data
   app.get('/api/users/:userId/financial-data', (req, res) => {
     const userId = req.params.userId;
     const userData = mockUsers[userId];
     
     if (!userData) {
       return res.status(404).json({ error: 'User not found' });
     }
     
     console.log(`Serving financial data for user: ${userId}`);
     res.json(userData);
   });

   // Get all available user profiles (for demo purposes)
   app.get('/api/users', (req, res) => {
     const userProfiles = Object.keys(mockUsers).map(userId => ({
       userId,
       name: mockUsers[userId].name,
       profile: mockUsers[userId].profile
     }));
     res.json(userProfiles);
   });

   app.listen(port, '0.0.0.0', () => {
     console.log(`Mock Banking Data API running on port ${port}`);
   });