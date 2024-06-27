import json
import sys
from json_minify import json_minify
import os
from pathlib import Path
import re
import string

NON_CONTENT_POS = ["AUX", "DET", "ADP", "SCONJ", "CONJ", "CCONJ", "PUNCT", "SYM", "X", "SPACE"]

def prepare_config_for_hf() -> dict:
    """
    Write minified config so huggingface can read it even if it has comments, and handle env variables
    """

    # Read non-minified config with comments
    config_file_path = sys.argv[-1]


    with open(config_file_path, "r") as f:
        minified_config = json_minify(f.read())
        config = json.loads(minified_config)

    # Handle env variables templates: {env:ENV_VARIABLE}
    for key, value in config.items():
        if isinstance(value, str):
            results = re.findall(r'{env:(.*)}', value)
            for env_variable in results:
                value = value.replace(f"${{env:{env_variable}}}", os.environ[env_variable])
            
            if any(results):
                config[key] = value
                
    # for distributed training:
    config["local_rank"]=int(os.environ.get("LOCAL_RANK", -1))
    # if len(sys.argv)>2:
    #     arg_name, arg_value = sys.argv[1].replace("--", "").replace("-", "_").split("=") # will convert '--local-rank=<number>' into ["local_rank", "<number>"] 
    #     config[arg_name] = arg_value

    # Save minified config
    new_config_path = f"{os.environ['TMPDIR']}/controlled_reduction/minified_configs/{config_file_path}"
    Path(os.path.dirname(new_config_path)).mkdir(parents=True, exist_ok=True)
    with open(new_config_path, "w") as f:
        f.write(json.dumps(config))

    sys.argv[-1] = new_config_path

    return config

import spacy
nlp = spacy.load("en_core_web_sm")

def filter_function_words(text: str) -> str:
    return " ".join([token.text for token in nlp(text) if token.text not in string.punctuation and token.pos_ not in NON_CONTENT_POS])
