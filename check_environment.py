"""აჩვენე Python გარემოს ძირითადი ინფორმაცია.

გაუშვი პროექტის საქაღალდიდან:
    python check_environment.py

აქ მხოლოდ Python-ის სტანდარტული ბიბლიოთეკა გვჭირდება.
"""

import platform
import sys
from pathlib import Path

print("Python executable:","1", sys.executable)
print("Python version:", "2", platform.python_version())
print("Architecture:", "3", platform.machine())
print("Working directory:", "4", Path.cwd())

# გარემოში sys.prefix მიუთითებს .venv-ზე, sys.base_prefix კი საწყის Python-ზე.
print("Virtual environment:", sys.prefix != sys.base_prefix)
