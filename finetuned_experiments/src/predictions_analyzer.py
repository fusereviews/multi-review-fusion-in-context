import os
import sys
from datasets import load_metric
import pandas as pd
import numpy as np    
import json
import re
from src.concatenate_highlights import concatenate_highlights


class PredictionsAnalyzer:
    """
    Extracts an analyzed result for each prediction instead of an aggregate of all predictions
    """

    def __init__(self, tokenizer, preprocessor, output_dir: str, rouge_metric, meteor_metric) -> None:
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.output_dir = output_dir
        self.rouge_metric = rouge_metric
        self.meteor_metric = meteor_metric



    def concatenate_highlighted_text(self, input, highlight_start_token, highlight_end_token, review_delim_token):
        pattern = fr'{highlight_start_token}(.*?){highlight_end_token}'
        return ". ".join([" ".join(re.findall(pattern, review, re.DOTALL)) for review in input.split(review_delim_token)])


    def write_predictions_to_file(self, predictions, dataset, df, is_tokenized=True):
        objects = self._clean_predictions(predictions, dataset, is_tokenized)

        # Calculate rouge and meteor between gold and summaries (if there is gold)
        if objects.get('gold') is not None:
            self.calculate_rouge_between_gold_n_prediction(objects, objects['predicted'], objects['gold'], prefix="gold")
            self.calculate_meteor_between_gold_n_prediction(objects, objects['predicted'], objects['gold'], prefix="gold")

        objects["database"] = list(df["database"])
        objects["subset"] = list(df["subset"])
        objects["summary_name"] = list(df["summary_name"])
        objects["alignments"] = list(df["alignments"])
        objects["clustered_alignments"] = list(df["clustered_alignments"])
        objects["full_inputs"] = [json.dumps({"alignments" : row["alignments"],
                                              "clustered_alignments" : row["clustered_alignments"],
                                              "review_0_text" : row["review_0_text"],
                                              "review_1_text" : row["review_1_text"],
                                              "review_2_text" : row["review_2_text"],
                                              "review_3_text" : row["review_3_text"],
                                              "review_4_text" : row["review_4_text"],
                                              "review_5_text" : row["review_5_text"],
                                              "review_6_text" : row["review_6_text"],
                                              "review_7_text" : row["review_7_text"],
                                              "review_0_highlight_spans" : row["review_0_highlight_spans"],
                                              "review_1_highlight_spans" : row["review_1_highlight_spans"],
                                              "review_2_highlight_spans" : row["review_2_highlight_spans"],
                                              "review_3_highlight_spans" : row["review_3_highlight_spans"],
                                              "review_4_highlight_spans" : row["review_4_highlight_spans"],
                                              "review_5_highlight_spans" : row["review_5_highlight_spans"],
                                              "review_6_highlight_spans" : row["review_6_highlight_spans"],
                                              "review_7_highlight_spans" : row["review_7_highlight_spans"]}) for i,row in df.iterrows()]


        # Calculate rouge and meteor between input and summary
        highlights_input = [self.concatenate_highlighted_text(input=self.preprocessor.preprocess_input(dict(row)), 
                                                              highlight_start_token=self.preprocessor.special_tokens_constants["highlight_start"], 
                                                              highlight_end_token=self.preprocessor.special_tokens_constants["highlight_end"], 
                                                              review_delim_token=self.preprocessor.special_tokens_constants["review_delim"]) for index, row in df.iterrows()]
        self.calculate_rouge_between_gold_n_prediction(objects, objects['predicted'], highlights_input, prefix="highlights")
        self.calculate_meteor_between_gold_n_prediction(objects, objects['predicted'], highlights_input, prefix="highlights")
        self._save_to_file(objects)

    def calculate_rouge_between_gold_n_prediction(self, objects, decoded_predictions, gold, prefix: str):
        result_per_pred = self.rouge_metric.compute(predictions=decoded_predictions, references=gold, use_stemmer=True, use_aggregator=False)
        objects[f'{prefix}_rouge1'] = [x for x in result_per_pred['rouge1']]
        objects[f'{prefix}_rouge2'] = [x for x in result_per_pred['rouge2']]
        objects[f'{prefix}_rougeL'] = [x for x in result_per_pred['rougeL']]

    def calculate_meteor_between_gold_n_prediction(self, objects, decoded_predictions, gold, prefix: str):
        results = [self.meteor_metric.compute(predictions=[prediction], references=[gold[i]]) for i,prediction in enumerate(decoded_predictions)]
        objects[f'{prefix}_meteor'] = [round(result['meteor']*100, 4) for result in results]

    def _clean_predictions(self, predictions, dataset, is_tokenized):
        
        def remove_special_tokens(curr_preds):
            all_special_tkns = sum([special_tkns if type(special_tkns)==list else [special_tkns] for special_tkns in self.tokenizer.special_tokens_map.values()], [])
            start_summary_tkn = self.preprocessor.special_tokens_constants["is_summary"]
            curr_preds = [re.sub(r'|'.join(map(re.escape, all_special_tkns)), '', pred) for pred in curr_preds] # remove the special tokens
            return curr_preds


        def remove_pad_tokens(prediction_tokens):
            """
            We want to calculate the num of tokens without the padding
            """

            return [token for token in prediction_tokens if token != self.tokenizer.pad_token_id]

        # Non-tokenized can be outputs not from a model, such as naive concatenation
        if not is_tokenized:
            decoded_predictions = predictions
            input_seqs = None
            clean_input_seqs = dataset
            input_tokenizer_lengths = None
            predictions_tokenizer_lengths = None
        else:
            decoded_predictions = self.tokenizer.batch_decode(predictions)
            
            decoded_predictions = remove_special_tokens(decoded_predictions)
            decoded_predictions = [pred.strip() for pred in decoded_predictions]

            input_seqs = [self.tokenizer.decode(dataset[i]['input_ids'])
                            for i in range(len(dataset))]
            clean_input_seqs = [self.tokenizer.decode(dataset[i]['input_ids'], skip_special_tokens=True)
                            for i in range(len(dataset))]

            # Length can be useful to see if the model actually saw everything
            predictions_tokenizer_lengths = [len(remove_pad_tokens(predictions[i])) for i in range(len(predictions))]
            input_tokenizer_lengths = [len(dataset[i]['input_ids']) for i in range(len(dataset))]

        gold = None
        gold_tokenizer_lengths = None
        if 'labels' in dataset[0]:
            gold = [self.tokenizer.decode(dataset[i]['labels']) for i in range(len(dataset))]
            gold = remove_special_tokens(gold)
            # Length can be useful to see if the model actually saw everything
            gold_tokenizer_lengths = [len(dataset[i]['labels']) for i in range(len(dataset))]

        objects = {"input": input_seqs, "clean_input": clean_input_seqs, "input_tokenizer_length": input_tokenizer_lengths, "predicted": decoded_predictions, "prediction_tokenizer_length": predictions_tokenizer_lengths}
        if gold is not None:
            objects["gold"] = gold
            objects["gold_tokenizer_length"] = gold_tokenizer_lengths

        return objects

    def _save_to_file(self, objects):
        df = pd.DataFrame(objects)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        output_prediction_file = os.path.join(
            self.output_dir, "generated_predictions.csv")
        
        # reorder ("database", "subset" and "summary_name" should be first columns and "clustered_alignments" should be the last)
        first_columns = ["database", "subset", "summary_name"]
        last_columnes = ["full_inputs"]
        new_column_order = first_columns + [col for col in df.columns if col not in first_columns+last_columnes] + last_columnes
        df = df[new_column_order]
        
        # save dataframe
        df.to_csv(output_prediction_file, index=False)
