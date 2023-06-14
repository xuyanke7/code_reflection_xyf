import time


def rate_limited(interval):
    def decorator(func):
        last_time_called = [0.0]

        def wrapper(*args, **kwargs):
            elapsed = time.monotonic() - last_time_called[0]
            left_to_wait = interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            result = func(*args, **kwargs)
            last_time_called[0] = time.monotonic()

            return result
        return wrapper
    return decorator


if __name__ == "__main__":
    ...
