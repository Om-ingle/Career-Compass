# CareerCompass Local Development Start Script
Write-Host "Starting CareerCompass Backend Services..." -ForegroundColor Green

# Start Mock Data API
Write-Host "`nStarting Mock Data API on port 8081..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot\mock-data-api'; npm start"

# Give the mock API time to start
Start-Sleep -Seconds 3

# Start AI Agent
Write-Host "Starting AI Agent on port 8080..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot\ai-agent'; python main.py"

Write-Host "`nServices are starting in separate windows." -ForegroundColor Green
Write-Host "Mock Data API: http://localhost:8081" -ForegroundColor Cyan
Write-Host "AI Agent: http://localhost:8080" -ForegroundColor Cyan
Write-Host "`nTo test the services, run: python test-integration.py" -ForegroundColor Yellow