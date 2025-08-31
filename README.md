Cleanlify API with Cherrypy framework
========
This project features the designing and building the API for Cleanlify.


Steps to SETUP Project
================

Prequisites:
- Python 3.11.13
- Google Cloud Account (For Google Knowledge Graph)
- Google AI Studio Account (For Gemini)

1. **Prepare Code & Environment**
   
   - Clone the project repo and navigate to the root directory
     ```
     git clone https://github.com/tasemgt/cleanlify_api.git
     cd cleanlify_api
     ```
   - Open in vscode for better coding management & experience. We will remain in this project directory and run our code from here.
   - Create a virtual environment (To manage versions of libraries)
     ```
     python -m venv .venv
     ```
   - Activate virtual environment if not active already
     ```
     source .venv/bin/activate
     ```
   - You can confirm that you are in the correct environment with the command below:
     This should return a path that ends with `/cleanlify_api/.venv/bin/python`
     ```
     which python
     ```
   - Install packages/libraries (CherryPy, pandas, pyspellchecker, etc...) with the command
     ```
     pip install -r requirements.txt
     ```
   - Create Database and Tables (To store users and clean summaries)
     ```
     python db.py
     ```
   - Start the application.
     If successful, you should see "Serving on http://127.0.0.1:8080 ENGINE Bus STARTED" in the console. (Do not close the terminal as this is our server running)
     ```
     python app.py
     ```
   - Open another terminal window and navigate to the same root directory i.e `/cleanlify_api`
   - Seed "users" into the database (This creates fake user accounts for the purpose of demonstrating functionality)
     ```
     python seed_users.py
     ```
   - Create a .env file to store your environment variables
     ```
     touch .env
     ```
     Open this file and paste the following:
     ```
     GEMINI_API_KEY = "fake_key_gemini" # Google Gemini API Key
     GOOGLE_KG_API_KEY = "fake_key_gkg" # Google Knowledge Graph API Key
     GOOGLE_KG_API_URL = "https://kgsearch.googleapis.com/v1/entities:search" # Google Knowledge Graph API URL
     ```
     
2. **Google Cloud GCP (For Google Knowledge Graph)**
   - Create an account in GCP
   - Create Project on gcp
   - Search for 'Knowledge Graph Search API' and Enable it
   - From side menu, locate and select 'API and Services'
   - Select 'Enabled API and services' to confirm that the enabled 'Knowledge Graph Search API' service is on the list
   - From same menu, select 'Credentials', click on 'Create credentials' and select 'API key'. This would create an API key to use in the application.
   - Copy the API key and replace the `fake_key_gkg` in the .env file created earlier.
   - This now gives you access to GKG to match entities and generate suggestions.
  
3.  **Google AI Studio (For Google Gemini LLM access)**
   - Go to `https://aistudio.google.com`, sign in if not already.
   - Find and select `Get API Key`
   - Select `Create API Key`
   - This might create in an existing or new project if it is the first time.
   - Copy API Key and replace the `fake_key_gemini` in the .env file created earlier.

 NB: Your app will still run without these keys but you will not be able to use Google KG, or LLM to get cluster suggestions in the application.

4. If you get to this point, your backend API is completely set up and is ready to receive requests from the Front end appplication.



Contact
=======
With ❤️ : [Michael Tase](https://www.linkedin.com/in/michael-tase-4151216a)

