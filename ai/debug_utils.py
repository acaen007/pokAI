# put at top of your module
VERBOSE = False

def debug_print(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

# then everywhere
# print(f"…")   →   debug_print(f"…")