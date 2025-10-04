import logging
import os

from src.factory import NoiseFactory
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

def main():
    print("Hello from thesis!")


if __name__ == "__main__":
    main()
