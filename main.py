import os
import sys
import argparse
import subprocess

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def run_cleaner():
    print("--- Running Data Cleaner ---")
    from src.processing import cleaner
    cleaner.process_pipeline()

def run_chunker():
    print("--- Running Chunker ---")
    from src.processing import chunker
    chunker.main()

def run_build_db():
    print("--- Building Pinecone Vector Database ---")
    from src.rag_engine import build_db
    build_db.build_db()

def run_dashboard(port=8501):
    print("==================================================")
    print("📈 AI-Powered Market Trend & Sentiment Forecaster")
    print("==================================================")
    print(f"Starting Streamlit dashboard on port {port}...")
    app_path = os.path.join(os.path.dirname(__file__), "src", "ui", "app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", app_path, "--server.port", str(port)]
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="AI-Powered Market Trend & Consumer Sentiment Forecaster")
    parser.add_argument("--clean", action="store_true", help="Run the data cleaning pipeline")
    parser.add_argument("--chunk", action="store_true", help="Run the text chunker pipeline")
    parser.add_argument("--build-db", action="store_true", help="Build and upload the Pinecone vector index")
    parser.add_argument("--port", type=int, default=8501, help="Port to run Streamlit on (default: 8501)")
    
    args = parser.parse_args()

    if args.clean:
        run_cleaner()
    elif args.chunk:
        run_chunker()
    elif args.build_db:
        run_build_db()
    else:
        run_dashboard(port=args.port)

if __name__ == "__main__":
    main()
