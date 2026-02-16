# game_engine/utils/validators.py

from functools import wraps
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)

def accept_flexible_ids(func: Callable) -> Callable:
    """
    Decorator to accept both string and integer IDs in function arguments.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Get the function's signature
            import inspect
            sig = inspect.signature(func)
            
            # Get parameter names
            param_names = list(sig.parameters.keys())
            
            # Check if this is a method (has 'self' or 'cls')
            is_method = len(args) > 0 and param_names[0] in ['self', 'cls']
            
            # Normalize IDs in keyword arguments
            for key, value in kwargs.items():
                if 'id' in key.lower() or 'slip' in key.lower():
                    if value is not None:
                        kwargs[key] = str(value)
            
            # Normalize IDs in positional arguments (skip self/cls for methods)
            start_idx = 1 if is_method and len(args) > 0 else 0
            new_args = list(args)
            
            for i in range(start_idx, min(len(new_args), len(param_names))):
                param_name = param_names[i]
                if 'id' in param_name.lower() or 'slip' in param_name.lower():
                    if new_args[i] is not None:
                        new_args[i] = str(new_args[i])
            
            return func(*new_args, **kwargs)
            
        except Exception as e:
            logger.error(f"Flexible ID decorator failed: {e}")
            # Fall back to original function
            return func(*args, **kwargs)
    
    return wrapper