"""
Rate limiter — single source of the slowapi Limiter instance.

Imported by main.py (to attach to app.state) and by route modules
(to apply @limiter.limit decorators).
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
