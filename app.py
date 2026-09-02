import os
import sys

# Redirect app.py execution to main.py
main_file = os.path.join(os.path.dirname(__file__), 'main.py')
with open(main_file, 'r', encoding='utf-8') as f:
    code = f.read()

exec(code)
