def assertOrThrow(condition, exception: Exception):
    if not condition: raise exception
