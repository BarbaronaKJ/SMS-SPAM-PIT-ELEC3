"""
Error Handler Middleware
Centralized error handling
"""

from flask import jsonify
from functools import wraps

def handle_errors(f):
    """
    Decorator to handle errors in route handlers
    
    Usage:
        @app.route('/endpoint')
        @handle_errors
        def endpoint():
            ...
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            return jsonify({
                'error': str(e)
            }), 400
        except Exception as e:
            return jsonify({
                'error': f'An error occurred: {str(e)}'
            }), 500
    return decorated_function
