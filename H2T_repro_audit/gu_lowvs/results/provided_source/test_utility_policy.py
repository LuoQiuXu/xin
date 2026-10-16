"""CPU-only pure policy checks; run before touching GPU or datasets."""
from utility_policy import test_policy

if __name__ == "__main__":
    test_policy()
