FROM python:3.9-slim

# Set environment variables
ENV TOKEN=$TOKEN

# Install dependencies
RUN pip install -r requirements.txt

# Copy the bot code
COPY . /app

# Run the bot
CMD ["python", "bot.py"]
