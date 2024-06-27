from collections import defaultdict
from typing import List, Tuple
import pandas as pd
import json
import re
from src.concatenate_highlights import combine_text_parts_to_str, concatenate_highlights_row, merge_overlapping_intervals
import math

IOU_THR = 0.0001

class Preprocessor:
    """
    Preprocess inputs and outputs
    """

    def __init__(self, prefix, special_tokens_constants, should_add_highlights: bool = True, should_add_reviews_delim: bool = True, only_sents_with_highlights: bool = False, keep_only_highlights: bool = False):
        self.prefix = prefix
        self.special_tokens_constants = special_tokens_constants
        self.should_add_highlights = should_add_highlights
        self.should_add_reviews_delim = should_add_reviews_delim
        self.only_sents_with_highlights = only_sents_with_highlights
        self.keep_only_highlights = keep_only_highlights


    def concatenate_highlighted_text(self, reviews):
        pattern = fr'{self.special_tokens_constants["highlight_start"]}(.*?){self.special_tokens_constants["highlight_end"]}'
        return ". ".join([" ".join(re.findall(pattern, review, re.DOTALL)) for review in reviews])




    def preprocess_input(self, data_dict) -> str:
        """
        Converts input to str
        """
        reviews_names = list(set([key.replace("_text", "").replace("_highlight_spans", "") for key in data_dict.keys() if key.startswith("review")]))
        reviews_names.sort()
        
        final_reviews = []

        for reviews_name in reviews_names:
            curr_review_text = data_dict[f"{reviews_name}_text"]
            curr_highlighted_spans = data_dict[f"{reviews_name}_highlight_spans"]
            
            # Collect all indices of tokens that need to be added
            idx_to_tokens = defaultdict(list)

            if self.only_sents_with_highlights:
                text_parts = concatenate_highlights_row({
                    "doc_text": curr_review_text,
                    "highlight_spans": curr_highlighted_spans
                }, keep_full_sentences=True, return_str=False)

                for text_part in text_parts:
                    if text_part.is_highlight:
                        text_part.prefix = self.special_tokens_constants['highlight_start']
                        text_part.postfix = self.special_tokens_constants['highlight_end']
                curr_final_review = combine_text_parts_to_str(text_parts, keep_full_sentences=True)
                final_reviews.append(curr_final_review)
            else:
                if not self.should_add_highlights and not self.keep_only_highlights:
                    curr_highlighted_spans = []
                else:
                    if isinstance(curr_highlighted_spans, str):
                        curr_highlighted_spans = json.loads(curr_highlighted_spans)

                    # We don't care about nested highlights / consecutive highlights
                    curr_highlighted_spans = merge_overlapping_intervals(curr_highlighted_spans)

                    for start, end in curr_highlighted_spans:
                        idx_to_tokens[start].append(self.special_tokens_constants['highlight_start'])
                        idx_to_tokens[end].append(self.special_tokens_constants['highlight_end'])

                # Build concatenated text by running over the text in parts
                source_text_with_highlighted_spans = ""
                last_idx = 0
                for idx in sorted(idx_to_tokens.keys()):
                    # Take text up to the current point
                    source_text_with_highlighted_spans += curr_review_text[last_idx:idx]

                    # Add the necessary tokens
                    tokens = idx_to_tokens[idx]
                    for token in tokens:
                        source_text_with_highlighted_spans += token
                    last_idx = idx

                source_text_with_highlighted_spans += curr_review_text[last_idx:]
                final_reviews.append(source_text_with_highlighted_spans)

        if self.keep_only_highlights:
            final_text = self.concatenate_highlighted_text(final_reviews)
        elif self.should_add_reviews_delim: # add delimeter between reviews
            final_text = f" {self.special_tokens_constants['review_delim']} ".join(final_reviews)         
        else: # otherwise, add only a space between them
            final_text = " ".join(final_reviews)
        
        # Return text with prefix
        return f"{self.prefix} {final_text}".strip()


    def preprocess_output(self, summary_text, curr_input) -> str:
        """
        Converts output to str
        """
        return summary_text

def get_special_tokens_constants(is_t5_model: bool) -> dict:
    """
    Constants used for preprocessing input and output
    """

    special_tokens_constants = {}
    if is_t5_model:
        # T5 model has 100 special tokens by default
        special_tokens_constants['highlight_start'] = "<extra_id_1>"
        special_tokens_constants['highlight_end'] = "<extra_id_2>"
        special_tokens_constants['review_delim'] = "<extra_id_3>"
        special_tokens_constants['is_concat'] = "<extra_id_4>"
        special_tokens_constants['is_summary'] = "<extra_id_5>"
        # special_tokens_constants['highlight_delim'] = "<extra_id_6>"
    else:
        special_tokens_constants['highlight_start'] = "<highlight_start>"
        special_tokens_constants['highlight_end'] = "<highlight_end>"
        special_tokens_constants['review_delim'] = "<review_delim>"

        special_tokens_constants['is_concat'] = "<is_concat>"
        special_tokens_constants['is_summary'] = "<is_summary>"
        # special_tokens_constants['highlight_delim'] = "<highlight_delim>"

    return special_tokens_constants


def convert_row_spans_str_to_list_of_highlights(spans_str) -> List[Tuple[int, int]]:
    """
    A single row's spans string can have spaces and be non-continuous. Example: "5361, 5374;5380, 5446"
    """

    highlights = []
    if type(spans_str) != str and math.isnan(spans_str): # no alignments in this review-summary pair
        return []
    start_end_strs = spans_str.split(";")
    for start_end_str in start_end_strs:
        split = start_end_str.split(",")
        start = int(split[0].strip())
        end = int(split[1].strip())
        highlights.append((start, end))

    return highlights

def convert_highlight_rows_to_document_highlights(doc_reader, highlight_rows: pd.DataFrame) -> List[List[Tuple[str, str, list]]]:
    """
    Convert from multiple highlight rows (csv) to document highlights
    """

    def handle_document_rows(doc_rows):

        any_row = doc_rows.iloc[0]
        any_row['database']
        curr_database = any_row['database']
        curr_subset = any_row['subset']
        curr_summary_name = any_row['summary_name']

        summary_text = doc_reader.read_summary(curr_database, curr_subset, curr_summary_name)


        output_csv_row_dict = {"database" : curr_database,
                               "subset" : curr_subset,
                               "summary_name" : curr_summary_name,
                               "summary_text" : summary_text}


        review_names = list(set(doc_rows['review_name']))
        review_names.sort()

        for review_name in review_names:
            curr_review_text = doc_reader.read_review(curr_database, curr_subset, curr_summary_name, review_name)
            curr_highlight_spans = doc_rows.loc[doc_rows["review_name"]==review_name]['review_span_offsets'].apply(convert_row_spans_str_to_list_of_highlights)
            curr_flattened_highlight_spans = [span for spans in curr_highlight_spans.to_list() for span in spans]
            output_csv_row_dict.update({f"{review_name}_text" : curr_review_text,
                                        f"{review_name}_highlight_spans" : curr_flattened_highlight_spans})
        # add the alignments for the coverage metric
        output_csv_row_dict["alignments"] = json.dumps([{"review_name":row['review_name'], "review_span_offsets":row["review_span_offsets"], "summary_span_offsets":row["summary_span_offsets"]} for index, row in doc_rows.iterrows()])
        output_csv_row_dict["clustered_alignments"] = json.dumps(get_clustered_alignments(doc_reader.data_json[curr_database][curr_subset][curr_summary_name]))
        return [output_csv_row_dict]

    def get_token_list(curr_tkn_ids, tokenization, only_content_tokens):

        def is_content_tkn(tkn_dict):
            return not tkn_dict['is_stop'] and not tkn_dict['pos'] in ['DET', 'PUNCT', 'SPACE', 'AUX']

        curr_tkn_ids_lst = [tkn_id for span in curr_tkn_ids for tkn_id in range(span[0], span[1]+1)]
        content_tkn_ids = [tkn["tkn_id"] for tkn in tokenization['summary_tkns'] if tkn["tkn_id"] in curr_tkn_ids_lst and (is_content_tkn(tkn) or not only_content_tokens)]
        return set(content_tkn_ids)

    def get_clustered_alignments(curr_instance):
        all_alignments = [{"review_name":review_instance["review_name"], "summary_alignment":alignment_instance["summary"], "review_alignment":alignment_instance["review"], "summary_alignment_content_tkns":get_token_list(alignment_instance["summary"]["tkn_id"], curr_instance['metadata']['tokenization'], True), "summary_alignment_all_tkns":get_token_list(alignment_instance["summary"]["tkn_id"], curr_instance['metadata']['tokenization'], False)} for review_instance in curr_instance["alignments"] for alignment_instance in review_instance["alignments"]]
        clustered_alignments = []
        # cluster alignments
        for align in all_alignments:
            # sometimes the alignments on the summary side mostly consist of words deemed stop words (e.g., "It is a must-see"), so summary_alignment_content_tkns is empty            
            if not align['summary_alignment_content_tkns']:
                subset_summary_alignment = [i for i,cluster in enumerate(clustered_alignments) if any(align["summary_alignment_all_tkns"].issubset(elem["summary_alignment_all_tkns"]) for elem in cluster)]
                if subset_summary_alignment: # found a summary-side alignment that consisted solely of non-content tokens which includes the current alignment
                    clustered_alignments[subset_summary_alignment[0]].append(align)
                else:
                    clustered_alignments.append([align])
                continue
            added_to_cluster = False
            for i,clustered_alignment in enumerate(clustered_alignments):
                if any(len(c_align['summary_alignment_content_tkns'].intersection(align['summary_alignment_content_tkns']))/len(align['summary_alignment_content_tkns'])>IOU_THR for c_align in clustered_alignment):
                    clustered_alignments[i].append(align)
                    added_to_cluster = True
                    break
            if not added_to_cluster:
                clustered_alignments.append([align])
        return [[{"review_name":inst["review_name"], "review_span_offsets": inst["review_alignment"]["idx"], "review_span_text": inst["review_alignment"]["text"], "summary_span_offsets": inst["summary_alignment"]["idx"], "summary_span_text": inst["summary_alignment"]["text"]} for inst in cluster] for cluster in clustered_alignments]



    document_highlights_df = highlight_rows.groupby(['database', 'subset', 'summary_name']).apply(handle_document_rows)
    # Flatten list of lists to a list
    return [document_highlight for document_highlights in document_highlights_df.to_list() for document_highlight in document_highlights]


