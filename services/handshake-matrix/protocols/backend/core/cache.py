```json
{
  "cache_code": """# backend/core/cache.py
from typing import Any, Optional, List, Union
import json
import redis
from datetime import timedelta
import logging
from functools import wraps

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, redis_host: str = 'localhost', 
                 redis_port: int = 6379,
                 default_ttl: int = 3600):
        self.client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )
        self.default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        \"\"\"Fetch value from cache\"\"\"
        try:
            value = self.client.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
            return None

    def set(self, key: str, value: Any, 
            ttl: Optional[int] = None,
            tags: Optional[List[str]] = None) -> bool:
        \"\"\"Store value in cache with optional TTL and tags\"\"\"
        try:
            ttl = ttl or self.default_ttl
            serialized = json.dumps(value)
            
            # Store value with TTL
            self.client.setex(key, ttl, serialized)
            
            # Store tag associations if provided
            if tags:
                for tag in tags:
                    self.client.sadd(f"tag:{tag}", key)
            return True
        except Exception as e:
            logger.error(f"Cache set error: {str(e)}")
            return False

    def invalidate(self, key_or_tag: str, 
                  is_tag: bool = False) -> bool:
        \"\"\"Invalidate cache by key or tag\"\"\"
        try:
            if is_tag:
                # Get all keys for tag
                tag_key = f"tag:{key_or_tag}"
                keys = self.client.smembers(tag_key)
                
                # Delete all keys and tag set
                if keys:
                    self.client.delete(*keys)
                self.client.delete(tag_key)
            else:
                self.client.delete(key_or_tag)
            return True
        except Exception as e:
            logger.error(f"Cache invalidation error: {str(e)}")
            return False

    def cached(self, ttl: Optional[int] = None,
              tags: Optional[List[str]] = None):
        \"\"\"Decorator for caching function results\"\"\"
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key from function name and arguments
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
                cache_key = ":".join(key_parts)

                # Try to get from cache
                cached_value = self.get(cache_key)
                if cached_value is not None:
                    return cached_value

                # If not in cache, execute function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl, tags)
                return result
            return wrapper
        return decorator""",

  "cache_integration_guidance": """### Integrating the Cache Layer

1. **Service Layer Integration**

```python
# Example integration in program_service.py
from core.cache import CacheManager

class ProgramService:
    def __init__(self):
        self.cache = CacheManager()

    @cache.cached(ttl=3600, tags=['programs'])
    def get_program(self, program_id: str):
        # Existing logic to fetch program
        pass

    def update_program(self, program_id: str, data: dict):
        # Update program
        result = self._do_update(program_id, data)
        # Invalidate cache
        self.cache.invalidate('programs', is_tag=True)
        return result
```

2. **API Endpoint Integration**

```python
# Example FastAPI endpoint with caching
from fastapi import APIRouter, Depends
from core.cache import CacheManager

router = APIRouter()
cache = CacheManager()

@router.get("/programs/{program_id}")
@cache.cached(ttl=1800, tags=['programs'])
async def get_program(program_id: str):
    # Existing endpoint logic
    pass
```

3. **Best Practices**

- Use consistent tag naming across related endpoints
- Set appropriate TTLs based on data volatility
- Invalidate cache on write operations
- Handle cache failures gracefully
- Monitor cache hit/miss rates""",

  "index_config_schema": """{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "indexes": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {
            "type": "string",
            "description": "Name of the index"
          },
          "collection": {
            "type": "string",
            "description": "Collection/table to index"
          },
          "fields": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "field": {
                  "type": "string",
                  "description": "Field name to index"
                },
                "type": {
                  "type": "string",
                  "enum": ["ascending", "descending", "text", "hashed"],
                  "description": "Type of index"
                },
                "weight": {
                  "type": "integer",
                  "minimum": 1,
                  "description": "Text index weight (optional)"
                }
              },
              "required": ["field", "type"]
            }
          },
          "options": {
            "type": "object",
            "properties": {
              "unique": {
                "type": "boolean",
                "default": false
              },
              "sparse": {
                "type": "boolean",
                "default": false
              },
              "expireAfterSeconds": {
                "type": "integer",
                "minimum": 0
              }
            }
          }
        },
        "required": ["name", "collection", "fields"]
      }
    }
  },
  "required": ["indexes"]
}""",

  "execution_logs": {
    "run_code_output": "No code execution needed - implementation code and schema generated successfully"
  }
}
```

The generated solution provides:

1. A robust CacheManager class that:
- Handles Redis interactions with error handling
- Supports TTL and tag-based caching
- Includes a decorator for easy function result caching
- Provides methods for cache invalidation

2. Integration guidance showing:
- How to integrate with services
- How to use with FastAPI endpoints
- Best practices for cache management

3. A comprehensive JSON schema for index configuration that:
- Supports multiple index types
- Allows for complex index configurations
- Includes options for unique/sparse indexes
- Supports text index weights

The implementation follows best practices for caching and provides flexibility for different use cases.