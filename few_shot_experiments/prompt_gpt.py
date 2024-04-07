import argparse
from datasets import load_dataset
import os
from pathlib import Path
import pandas as pd
from utils import *


def main(args):
    openai.api_key = os.getenv("OPENAI_API_KEY") if not args.openAI_key else args.openAI_key
    dataset = load_dataset("lovodkin93/FuseReviews", split=args.split)

    # create outdir if doesn't exist
    path = Path(args.outdir)
    path.mkdir(parents=True, exist_ok=True)
    
    with open("prompt.json", 'r') as f1:
        prompt_dict = json.loads(f1.read())
    
    # resume genertion if crashed
    if args.resume_file:
        already_generated_outputs = pd.read_csv(args.resume_file)
        logging.info(f'resume generation from instance: {len(already_generated_outputs)}')

    all_prompts_reg, all_prompts_CoT = generate_prompts(dataset=dataset, n_demos=args.n_demos, prompt_dict=prompt_dict)
    responses = already_generated_outputs["generated_responses"].tolist() if args.resume_file else []
    for prompt_i in range(len(all_prompts_reg)):
        if args.resume_file and prompt_i < len(already_generated_outputs): # skip already generated outputs if resumed generation
            continue
        curr_prompt = all_prompts_CoT[prompt_i] if args.CoT else all_prompts_reg[prompt_i]
        model_response = send_request(curr_prompt, args.model_name)
        if args.CoT:
            if "So, the answer is:" in model_response:
                model_response = model_response[model_response.index("So, the answer is:"):].replace("So, the answer is:", "")
            else: # model didn't follow the demonstrations' structure and didn't generate the ginal summary
                model_response = send_request(all_prompts_reg[prompt_i], args.model_name)
        responses.append(model_response.strip())
        if (prompt_i+1) % args.save_interval == 0:
            responses_df = pd.DataFrame({"generated_responses":responses})
            responses_df.to_csv(os.path.join(args.outdir, "generated_responses.csv"), index=False)
    
    # save results
    responses_df = pd.DataFrame({"generated_responses":responses})
    responses_df.to_csv(os.path.join(args.outdir, "generated_responses.csv"), index=False)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument('-o', '--outdir', type=str, default="outputs", help='path to output, where the csv with the responses will be saved.')
    argparser.add_argument('--openAI-key', type=str, default=None, help='API key of OpenAI.')
    argparser.add_argument('--split', type=str, default="test", help='data split (train, validation, or test). Default: test')
    argparser.add_argument('--n-demos', type=int, default=1, help='number of demonstrations. Default: 1')
    argparser.add_argument('--model-name', type=str, default="gpt-4", help='model name')
    argparser.add_argument('--CoT', action='store_true', default=False, help='whether to use CoT')
    argparser.add_argument('--save-interval', type=int, default=1, help='how often to save the results')
    argparser.add_argument('--resume-file', type=str, default=None, help='file to resume writing to if crashed.')
    args = argparser.parse_args()
    main(args)












