"""~/.kaggle/kaggle.json에서 username을 읽는다."""
import json
import os
import sys

p = os.path.expanduser("~/.kaggle/kaggle.json")
if not os.path.exists(p):
    sys.exit("[!] ~/.kaggle/kaggle.json 없음. Kaggle > Settings > Create New Token 후 배치할 것.")
print(json.load(open(p))["username"])
