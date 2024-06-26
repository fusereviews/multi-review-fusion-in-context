from src.run import main as seq_to_seq_main
from src.utils import prepare_config_for_hf


if __name__ == "__main__":

    config = prepare_config_for_hf()

    experiment_type = config['experiment_type']

    if experiment_type == "seq2seq":
        seq_to_seq_main()