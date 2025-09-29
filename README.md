## ThreatWatch Backend

Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```


Install the required dependencies
```bash
pip install -r requirements.txt
```


To Start the Backend Server
```bash
uvicorn main:app --reload
```