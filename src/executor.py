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
        
        Args:
            code (str): The Python code to execute
            test_code (str): Test assertions to validate the code
        
        Returns:
            Tuple[bool, str, float]: (success, error_message, reward)
        """
        temp_file = None
        
        try:
            # Combine code and test
            full_code = code + "\n\n" + test_code
            
            # Write to temporary file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False
            ) as f:
                f.write(full_code)
                temp_file = f.name
            
            # Execute with subprocess
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            # Check results and assign reward
            if result.returncode == 0 and 'PASS' in result.stdout:
                return True, None, 10
            elif 'SyntaxError' in result.stderr:
                return False, result.stderr, -10
            else:
                error_msg = result.stderr if result.stderr else result.stdout
                return False, error_msg, -0.5
                
        except subprocess.TimeoutExpired:
            return False, "Timeout: Code took too long to execute", -1.0
            
        except Exception as e:
            return False, f"Unexpected error: {str(e)}", -1.0
            
        finally:
            # Clean up temp file
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)


# Test the class
if __name__ == "__main__":
    executor = CodeExecutor(timeout=5)
    code = "def add(a, b):\n    return a + b"
    test = "assert add(2, 3) == 5\nprint('PASS')"
    success, error, reward = executor.execute(code, test)
    print(f"Reward: {reward}")