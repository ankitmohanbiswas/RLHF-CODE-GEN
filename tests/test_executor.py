from src.executor import CodeExecutor

executor = CodeExecutor(timeout=5)

TEST_CODE = """
assert add(2,3) == 5
assert add(0,0) == 0
assert add(-1,1) == 0
print("PASS")
"""

def test_correct_code():
    code = """
def add(a, b):
    return a + b
"""
    success, error, reward = executor.execute(code=code, test_code=TEST_CODE)

    assert success is True
    assert error is None
    assert reward == 1.0


def test_wrong_logic():
    code = """
def add(a, b):
    return a * b
"""
    success, error, reward = executor.execute(code=code, test_code=TEST_CODE)

    assert success is False
    assert reward == -0.5


def test_syntax_error():
    code = """
def add(a, b)
    return a * b
"""
    success, error, reward = executor.execute(code=code, test_code=TEST_CODE)

    assert success is False
    assert reward == -1.0
    assert "SyntaxError" in (error or "")


def test_infinite_loop():
    code = """
def add(a, b):
    while True:
        pass
"""
    success, error, reward = executor.execute(code=code, test_code=TEST_CODE)

    assert success is False
    assert reward == -1.0
    assert "Timeout" in (error or "")


def test_runtime_error():
    code = """
def add(a, b):
    return 0 / 0
"""
    success, error, reward = executor.execute(code=code, test_code=TEST_CODE)

    assert success is False
    assert reward == -0.5
