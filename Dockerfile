# Use a base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file (if you have one)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port (adjust if needed)
EXPOSE 8000

# Command to run the application
CMD ["python", "app.py"]