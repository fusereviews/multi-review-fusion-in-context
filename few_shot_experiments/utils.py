from transformers import AutoTokenizer
import time
import json
import openai
import logging
logging.basicConfig(level=logging.INFO)
import re
import numpy as np

tokenizer = AutoTokenizer.from_pretrained('gpt2', model_max_length=4096)
TOTAL_REVIEWS = 8

def send_request(content, model_name):
    if model_name == "gpt-3.5-turbo" and tokenizer.encode(content, max_length=4097, return_tensors="pt").shape[-1] > 4097:
        return "Prompt Too Long!"
    try:
        time.sleep(15) 
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": content}
            ]
        )
    except Exception as e:
        logging.error('Failed to send request: '+ str(e)) 
        return f"ERROR: {str(e)}"

    return response["choices"][0]["message"]["content"]

def extract_highlights(highlighted_rev, highlight_start_tkn, highlight_end_tkn):
    pattern = fr'{highlight_start_tkn}(.*?){highlight_end_tkn}'
    return re.findall(pattern, highlighted_rev, re.DOTALL)

def make_demo(instance, prompt_dict, with_CoT, is_test):
    instance_prompt = prompt_dict["prompt_structure"]    
    instance_prompt = instance_prompt.replace("{INST}", prompt_dict["instructions"]) # add instructions
    instance_prompt = instance_prompt.replace("{H_REV}", f"\n".join([f"{rev_name}: {rev}" for rev_name,rev in instance["reviews"].items()])) # add highlighted reviews
    instance_prompt = instance_prompt.replace("{HS}", prompt_dict['highlight_start']).replace("{HE}", prompt_dict['highlight_end'])

    # add list of highlights
    highlights_to_string_dict = {rev_name:"\n ".join([f"{i+1}. {highlight}" for i,highlight in enumerate(extract_highlights(highlighted_rev, "{HS}", "{HE}"))]) for rev_name,highlighted_rev in instance["reviews"].items()}
    highlights_to_string = "\n".join(f"{rev_name}:\n {highlight_lst}" for rev_name,highlight_lst in highlights_to_string_dict.items())
    instance_prompt = instance_prompt.replace("{HIGHLIGHTS}", highlights_to_string)

    # add CoT
    if with_CoT:
        if not is_test:
            alignments_to_string_lst = ["Spans " + " ; ".join([f'{",".join([str(i+1) for i in highlights])} ({rev_name})' for rev_name,highlights in alignment['alignments'].items() if highlights]) + f" are combined to form sentence {j+1}: {alignment['output_sentence']}" for j,alignment in enumerate(instance['highlights_output_alignment'])]
        alignments_to_string = "\n".join(alignments_to_string_lst) if not is_test else ""
        instance_prompt = instance_prompt.replace("{COT}", prompt_dict["cot_structure"])
        instance_prompt = instance_prompt.replace("{COT_STEPS}", alignments_to_string)
        if is_test:
            instance_prompt = instance_prompt.replace("So, the answer is: {A}", "")
    else: 
        instance_prompt = instance_prompt.replace("{COT}", "")

    # add answer for demonstrations
    answer = instance['output'] if not is_test else ""
    instance_prompt = instance_prompt.replace("{A}", answer)
    return instance_prompt.strip()

def add_highlights(text, highlights, highlight_start_tkn, highlight_end_tkn):
    if not highlights:
        return text
    highlights = sorted(highlights, key=lambda x: x[0])
    highlighted_text = text[:highlights[0][0]] # start with the text until first highlight

    for i,span in enumerate(highlights):
        end_idx_non_highlighted = highlights[i+1][0] if i<len(highlights)-1 else len(text) # if not final highlight - next non-highlighted span's end idx is the start of the next highlight, otherwise - it is the end of the doc
        addition_txt = highlight_start_tkn + text[span[0]:span[1]] + highlight_end_tkn + text[span[1]:end_idx_non_highlighted]
        highlighted_text += addition_txt
    
    # make sure the removal of the highlights yields the original text
    assert highlighted_text.replace(highlight_start_tkn, "").replace(highlight_end_tkn, "") == text
    return highlighted_text

def generate_prompts(dataset, n_demos, prompt_dict):
    train_ids = np.random.choice(len(prompt_dict["demos"]), n_demos, replace=False)
    used_demos = []
    head_prompt_reg, head_prompt_CoT = "", ""
    # add demonstrations
    for train_id in train_ids:
        train_item = prompt_dict["demos"][train_id]
        used_demos.append(train_item)

        # get regular prompt
        curr_prompt_demo_reg = make_demo(
            instance=train_item, prompt_dict=prompt_dict, with_CoT=False, 
            is_test=False
        )
        head_prompt_reg += curr_prompt_demo_reg
        head_prompt_reg += prompt_dict["demo_sep"]

        # get CoT prompt
        curr_prompt_demo_CoT = make_demo(
            instance=train_item, prompt_dict=prompt_dict, with_CoT=True, 
            is_test=False
        )
        head_prompt_CoT += curr_prompt_demo_CoT
        head_prompt_CoT += prompt_dict["demo_sep"]
    
    # add actual instances
    final_prompts_reg, final_prompts_CoT = [], []
    for instance in dataset:
        all_reviews = {f"review_{rev_i}":add_highlights(text=instance[f'review_{rev_i}_text'], 
                                                        highlights=json.loads(instance[f'review_{rev_i}_offsets']), 
                                                        highlight_start_tkn="{HS}", 
                                                        highlight_end_tkn="{HE}") for rev_i in range(TOTAL_REVIEWS)}
        # get regular prompt
        curr_inst_prompt_reg = make_demo(instance={"reviews":all_reviews}, prompt_dict=prompt_dict, with_CoT=False,
                                     is_test=True
        )
        full_prompt_reg = head_prompt_reg + curr_inst_prompt_reg
        final_prompts_reg.append(full_prompt_reg)

        # get CoT prompt
        curr_inst_prompt_CoT = make_demo(instance={"reviews":all_reviews}, prompt_dict=prompt_dict, with_CoT=True,
                                     is_test=True
        )
        full_prompt_CoT = head_prompt_CoT + curr_inst_prompt_CoT
        final_prompts_CoT.append(full_prompt_CoT)
    return final_prompts_reg, final_prompts_CoT