import subprocess
import webbrowser
import time
import sys
import os

def run_project():
    print("🚀 Starting EduGuard TN System...")

    # 1. Ensure the models exist before starting
    if not os.path.exists("models/regressor.pkl"):
        print("🧠 Model files not found. Training models first...")
        subprocess.run([sys.executable, "train_model.py"])
    
    # 2. Start the FastAPI Backend (main.py)
    # We use subprocess.Popen so it runs in the background
    print("📡 Starting Backend Server...")
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

    # 3. Wait a moment for the server to wake up
    time.sleep(3)

    # 4. Open the Web Page automatically
    # Get the absolute path to your index.html
    html_file = os.path.abspath("index.html")
    print(f"🌐 Opening Dashboard: {html_file}")
    
    webbrowser.open(f"file://{html_file}")

    print("\n✅ EduGuard is now running!")
    print("⚠️  Keep this window open to keep the system active.")
    print("⌨️  Press Ctrl+C to stop the server.")

    try:
        # Keep the script alive so the backend doesn't close
        backend_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping EduGuard...")
        backend_process.terminate()

if __name__ == "__main__":
    run_project()