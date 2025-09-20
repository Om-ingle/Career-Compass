FROM python:3.9-slim

   WORKDIR /app

   # Install dependencies
   COPY requirements.txt .
   RUN pip install -r requirements.txt

   # Copy source code
   COPY . .

   # Expose port
   EXPOSE 8080

   # Start the application
   CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]