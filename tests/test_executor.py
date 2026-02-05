from src.executor import CodeExecutor

executor=CodeExecutor(timeout=5)
print("=" * 60)
print("Testing has begun.....")
print("=" * 60)
code= """
def add (a,b):
    return a+b
    """
test="""

assert add(2,3) == 5
assert add(0,0) == 0
assert add(-1,1) == 0
print("PASS")
"""

#----------------------------------------------------------
#TEST 1
#-=-------------------------------------------------------
success, error, reward=executor.execute(code=code,test_code=test)
print(f"{success}")
print(f"Error message is {error}")
print(f"Reward for the system is {reward}")
assert success==True
assert reward==1.0
print("test 1 passed")

#----------------------------------------------------------
#TEST 2 LOGIC 
#-=-------------------------------------------------------
print("=" * 60)
print("2nd Testing has begun.....")
print("=" * 60)
w_code= """
def add (a,b):
    return a*b
    """

success, error, reward=executor.execute(code=w_code, test_code=test)
print(f"Success:{success}")
print(f"Error:{error}")
print(f"Reward:{reward}")

assert success==False
assert reward==-0.5
print("Test 2 Passed")
#----------------------------------------------------------
#TEST 3 SYNTAX 
#-=-------------------------------------------------------
print("=" * 60)
print("3rd Testing has begun.....")
print("=" * 60)
wrong_logic_code= """
def add (a,b)
    return a*b
    """

success, error, reward=executor.execute(code=wrong_logic_code, test_code=test)
print(f"Success:{success}")
print(f"Error:{error}")
print(f"Reward:{reward}")

assert success==False
assert reward==-1.0
assert "SyntaxError" in (error or "") 
print("Test 3 Passed")
#----------------------------------------------------------
#TEST 4 INFINITE LOOP
#-=-------------------------------------------------------
print("=" * 60)
print("4th Testing has begun.....")
print("=" * 60)
infinite_code= """
def add (a,b):
    while True:
        pass
    """
success, error, reward=executor.execute(code=infinite_code, test_code=test)
print(f"Success:{success}")
print(f"Error:{error}")
print(f"Reward:{reward}")

assert success==False
assert reward== -1.0
assert "Timeout" in (error or "")
print("Test 3 Passed")
#----------------------------------------------------------
#TEST 5 RUNTIME
#-=-------------------------------------------------------
print("=" * 60)
print("5th Testing has begun.....")
print("=" * 60)
w_code= """
def add (a,b):
    return 0/0
    """

success, error, reward=executor.execute(code=w_code, test_code=test)
print(f"Success:{success}")
print(f"Error:{error}")
print(f"Reward:{reward}")

assert success==False
assert reward==-0.5
print("Test 5 Passed")

print("=" * 60)
print(" ALL TESTS PASSED!")
print("=" * 60)
