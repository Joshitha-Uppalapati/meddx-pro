import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from meddx.serve.ui_app import main

if __name__ == "__main__":
    main()
