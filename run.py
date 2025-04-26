import uvicorn
from src.api import app

if __name__ == "__main__":
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload
        reload_dirs=["src"],  # Watch for changes in src directory
        reload_excludes=["*.pyc", "__pycache__"]  # Exclude these from triggering reload
    ) 