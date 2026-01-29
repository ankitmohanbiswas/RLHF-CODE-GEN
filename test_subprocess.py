import subprocess
import tempfile
import os

infinite_loop="""
while True:
    pass
"""
with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
    f.write('print("code from temp file")')
    path=f.name
    print(f"Temp file created at :{path}")

result=subprocess.run(['python', path],
                        capture_output=True,
                        text=True)

print(f"The output is {result.stdout}")
try:
    result=subprocess.run(['python', path],
                        capture_output=True,
                        text=True,
                        timeout=2)

    print(f"The output is {result.stdout}")
except subprocess.TimeoutExpired:
    print("code killed ---> took too long")
finally:
    os.unlink(path=path)
    print("The file has been deleted")


