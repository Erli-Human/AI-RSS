import subprocess
import sys
import os

def install_requirements():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def launch_app():
    subprocess.Popen([sys.executable, "app.py"])

if __name__ == "__main__":
    install_requirements()
    launch_app()