# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy necessary files
COPY model.py /app/
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the script
CMD ["python", "model.py"]