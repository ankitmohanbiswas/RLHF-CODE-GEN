"""
Code Executor Module for RLHF Training

This module provides a safe execution environment for running user-generated
Python code with automated testing and reward assignment.
"""

import subprocess
import tempfile
import os
from typing import Tuple


class CodeExecutor:
    """
    A safe executor for running Python code with test cases.
    
    This class executes user-generated code in isolation using temporary files
    and subprocess, with timeout protection against infinite loops.
    
    Attributes:
        timeout (int): Maximum seconds allowed for code execution
    
    Example:
        >>> executor = CodeExecutor(timeout=5)
        >>> code = "def add(a, b):\\n    return a + b"
        >>> test = "assert add(2, 3) == 5\\nprint('PASS')"
        >>> success, error, reward = executor.execute(code, test)
        >>> print(f"Reward: {reward}")
        Reward: 1.0
    """
    
    def __init__(self, timeout: int = 5):
        """
        Initialize the executor with specified timeout.
        
        Args:
            timeout (int): Maximum seconds to wait for code execution.
                          Default is 5 seconds.
        """
        self.timeout = timeout
    
    def execute(self, code: str, test_code: str) -> Tuple[bool, str, float]:
        """
        Execute Python code with test cases and return reward.
        
        This method runs the provided code in an isolated subprocess,
        evaluates it against test cases, and assigns a reward based on
        the execution outcome.
        
        Reward System:
            +1.0: Code executes successfully and passes all tests
            -0.5: Code runs but fails tests or has runtime errors
            -1.0: Code has syntax errors or times out (infinite loop)
        
        Args:
            code (str): The Python code to execute (e.g., function definition)
            test_code (str): Test assertions to validate the code.
                           Must include print('PASS') for successful execution.
        
        Returns:
            Tuple[bool, str, float]: A tuple containing:
                - success (bool): True if all tests passed, False otherwise
                - error_message (str): Error details if failed, None if success
                - reward (float): Numerical reward for RL training
        
        Example:
            >>> executor = CodeExecutor()
            >>> code = '''
            ... def reverse_string(s):
            ...     return s[::-1]
            ... '''
            >>> test = '''
            ... assert reverse_string("hello") == "olleh"
            ... assert reverse_string("") == ""
            ... print('PASS')
            ... '''
            >>> success, error, reward = executor.execute(code, test)
            >>> print(success, reward)
            True 1.0
        
        Raises:
            No exceptions are raised; all errors are caught and returned
            as part of the tuple response.
        """
        temp_file = None
        
        try:
            # Combine user code with test assertions
            full_code = code + "\n\n" + test_code
            
            # Write combined code to a temporary Python file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False
            ) as f:
                f.write(full_code)
                temp_file = f.name
            
            # Execute the code in a subprocess with timeout protection
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            # Evaluate execution results and assign reward
            if result.returncode == 0 and 'PASS' in result.stdout:
                # Perfect execution: code runs and passes all tests
                return True, None, 1.0
            
            elif 'SyntaxError' in result.stderr:
                # Syntax error: code is not valid Python
                return False, result.stderr, -1.0
            
            else:
                # Runtime error or failed assertion: code runs but fails
                error_msg = result.stderr if result.stderr else result.stdout
                return False, error_msg, -0.5
                
        except subprocess.TimeoutExpired:
            # Code execution exceeded timeout limit (likely infinite loop)
            return False, "Timeout: Code took too long to execute", -1.0
            
        except Exception as e:
            # Catch any unexpected errors (file system, permissions, etc.)
            return False, f"Unexpected error: {str(e)}", -1.0
            
        finally:
            # Always clean up temporary file to prevent accumulation
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)


def load_problems(path: str = "data/problems.json") -> list:
    """
    Load coding problems from JSON dataset.
    
    Args:
        path (str): Path to the JSON file containing problems.
                Default is "data/problems.json"
    
    Returns:
        list: List of problem dictionaries, each containing:
            - id: Problem identifier
            - prompt: Starting code for the problem
            - description: Human-readable problem description
            - test: Test code to validate solutions
    
    Example:
        >>> problems = load_problems()
        >>> print(problems[0]['description'])
        Add two numbers
    
    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    import json
    
    with open(path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    """
    Test the executor with a simple example.
    Run this file directly to verify the executor works correctly.
    """
    print("Testing CodeExecutor...")
    
    executor = CodeExecutor(timeout=2)
    
    # Test 1: Correct code
    problems = load_problems("data/problems.json")
    p= problems[0]
    print(f"testing problem {p['description']} at {0} index")
    code=p['prompt']
    test=p['test']

    success, error, reward=executor.execute(code, test_code=test)
    print(f"Success:{success}")
    print(f"Error:{error}")
    print(f"Reward:{reward}")
