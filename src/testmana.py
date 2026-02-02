from executor import CodeExecutor

executor = CodeExecutor(timeout=5)

# Your generated code
code = """def add(a,b):
    return a+b"""

# Your test
test = """assert add(5, 67) == 72
assert add(0, 0) == 0
print('PASS')"""

success, error, reward = executor.execute(code, test)

print(f"Success: {success}")
print(f"Reward: {reward}")
print(f"Error: {error}")