Ollama UBNT24: AI Narrative Analysis Server
This project provides a fully automated, containerized solution for deploying an AI narrative analysis server on a clean Ubuntu 24.04 installation. The system utilizes Ollama for local LLM inference, FastAPI as a web API, and PostgreSQL for data storage, all orchestrated with Docker Compose. It also includes the Open WebUI for a user-friendly interface to interact with your LLMs.

The primary goal of this project is to create a robust, reproducible, and portable deployment for AI research and data analysis.

Core Components
Ollama: A powerful tool for running large language models (LLMs) locally on your GPU.

FastAPI: A high-performance Python web framework for building the API endpoints.

PostgreSQL with pgvector: A robust relational database for storing data, including vector embeddings for AI analysis.

Open WebUI: A user-friendly web interface for interacting with your Ollama server and models.

Scraper Agent: A Python-based agent that scrapes articles, uses Ollama for narrative analysis, and stores the results in the database.

Installation and Setup
This project is designed to be deployed on a clean Ubuntu 24.04 machine. Follow these steps to get your server up and running.

Step 1: Copy Project Files
Copy all project files to the target Ubuntu machine. The directory structure should look like this:

ollamaubnt24/
├── agents/
├── ollama/
├── .env
├── docker-compose.yml
├── main.py
├── database.py
├── models.py
├── requirements.txt
├── deploy.sh
└── build_project_structure.sh

Step 2: Run the Deployment Script
This single script handles all system-level dependencies, including Docker, the NVIDIA Container Toolkit, and PostgreSQL.

Edit the deploy.sh script and replace "YOUR_STRONG_POSTGRES_PASSWORD" with a secure password.

Make the script executable:

chmod +x deploy.sh

Run the deployment script as root:

sudo ./deploy.sh

This script will install all dependencies, configure PostgreSQL, and set up your project environment.

Step 3: Start the Docker Services
After running the deployment script, all of the necessary services are ready to be started.

docker compose up -d --build

Step 4: Access the UI and API
Once the containers are running, you can access the user interface and the API.

Open WebUI: Navigate to http://<your_server_ip>:8080 in your web browser. This will give you an interface for interacting with your LLMs.

FastAPI API: The API documentation is available at http://<your_server_ip>:8000/docs. You can use curl or a tool like Postman to interact with your endpoints.

Usage
Running the Scraper Agent
The scraper agent is configured to run as a one-shot process. Once the Ollama models have been downloaded (check the Open WebUI or ollama logs), you can run the scraper to populate your database with data.

docker compose logs -f scraper_agent

Querying the API
You can use the curl command to query the API. Remember to replace YOUR_SECRET_API_KEY with the key you set in your .env file.

curl -X 'GET' \
  'http://localhost:8000/analyze_node/1' \
  -H 'X-Api-Key: YOUR_SECRET_API_KEY'

