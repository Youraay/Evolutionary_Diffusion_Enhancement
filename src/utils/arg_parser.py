import argparse

def args():

    parser = argparse.ArgumentParser(
        description="Führt den Genetischen Suchalgorithmus aus."
    )

    parser.add_argument(
        '--experiment_id', 
        type=str, 
        help='Id of the experiment'
    )
    args = parser.parse_args()

    return args