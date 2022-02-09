def assert_or_throw(condition, exception: Exception):
    if not condition:
        raise exception
