deGlitch - Device Vulnerability Detection System
===============================================

deGlitch is a web-based application that monitors the security risk of connected devices 
and alerts users about potential vulnerabilities. The system detects various types of 
attacks on devices such as mobile phones, laptops, headphones, and printers and provides 
an interface to take corrective actions.

-------------------------------------------------
FEATURES
-------------------------------------------------
- Real-time Risk Monitoring: Displays the security status of devices.
- Intuitive UI: Provides a user-friendly dashboard for viewing device status.
- Dynamic Risk Assessment: Analyzes vulnerability levels (Low, Medium, High).
- Fix Button: Suggests corrective measures for detected threats.
- API Integration: Connects to a backend server to fetch analysis results.

-------------------------------------------------
PROJECT STRUCTURE
-------------------------------------------------
deGlitch
├── deGlitch.html    - Main HTML file for the frontend
├── deGlitch.css     - CSS file for styling the frontend
├── app.py           - Flask server to serve the HTML and CSS
├── server.py        - Flask API server to provide risk analysis results
└── tmp
    └── result.txt   - Contains risk analysis results

-------------------------------------------------
INSTALLATION AND SETUP
-------------------------------------------------
1. Clone the Repository
   $ git clone https://github.com/your-username/deGlitch.git
   $ cd deGlitch

2. Install Dependencies
   $ pip install flask flask-cors

3. Run Flask Servers
   a) Run app.py (to serve frontend)
      $ python app.py
      Access the frontend at: http://localhost:8000

   b) Run server.py (to provide API data)
      $ python server.py
      API endpoint available at: http://localhost:50000/get_result

-------------------------------------------------
USAGE INSTRUCTIONS
-------------------------------------------------
1. Start Backend Servers: Run both app.py and server.py.
2. Open Frontend: Access http://localhost:8000 in your browser.
3. Check Device Status: Monitor device security status.
4. Analyze Results: Devices will display Low, Medium, or High Risk based on vulnerability analysis.
5. Fix Issues: Click the "Fix It" button for suggested actions.

-------------------------------------------------
CONFIGURATION
-------------------------------------------------
1. API Endpoint Update:
   If you need to change the API endpoint, modify the serverUrl in deGlitch.html:
   const serverUrl = 'http://172.16.44.211:50000/get_result';

2. Result File:
   Ensure that the result file result.txt is stored in:
   C:\Users\Akshay Prakash\Documents\Semester 4\Hackathon\tmp\result.txt

-------------------------------------------------
TROUBLESHOOTING
-------------------------------------------------
- API Not Responding: Ensure server.py is running.
- Result Not Found: Verify result.txt exists in the correct path.
- CORS Issues: Ensure Flask-CORS is properly configured in server.py.

-------------------------------------------------
CONTACT
-------------------------------------------------
For any queries or suggestions, feel free to contact:
- Developer: Akshay Prakash

Happy Coding!
