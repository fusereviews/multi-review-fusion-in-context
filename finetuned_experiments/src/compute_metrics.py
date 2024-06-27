import numpy as np

from src.utils import filter_function_words


def compute_rouge_metrics(predictions: list, references: list, metric, prefix: str, should_filter_function_words: bool = False) -> dict:
    assert len(predictions) == len(references)

    filtered_predictions = predictions
    filtered_references = references
    if should_filter_function_words:
        filtered_predictions = []
        for prediction in predictions:
            filtered_predictions.append(filter_function_words(prediction))

        filtered_references = []
        for reference in references:
            filtered_references.append(filter_function_words(reference))
    result = metric.compute(predictions=filtered_predictions,
                            references=filtered_references, use_stemmer=True)
    # Extract a few results from ROUGE
    result_parsed = {f"{prefix}_{key}": round(value*100, 4) for key, value in result.items()}

    # Add also precision and recall
    # result_parsed.update({f"{prefix}_{key}_precision": value.mid.precision * 100 for key, value in result.items()})
    # result_parsed.update({f"{prefix}_{key}_recall": value.mid.recall * 100 for key, value in result.items()})

    result_parsed = {k: round(v, 4) for k, v in result_parsed.items()}

    return result_parsed

def compute_meteor_metrics(predictions: list, references: list, metric, prefix: str) -> dict:
    result = metric.compute(predictions=predictions, references=references)
    return {
        f"{prefix}meteor": round(result['meteor']*100, 4)
    }


def compute_highlights_metrics(predictions: list, reviews: list, clustered_reviews_alignments: list, full_inputs: list, metric_faithfulness, metric_coverage, prefix: str) -> dict:
    result_faithfulness = metric_faithfulness.evaluate(predictions=predictions, full_inputs=full_inputs)
    result_coverage = metric_coverage.evaluate(predictions=predictions, clustered_reviews_alignments=clustered_reviews_alignments)

    faithfulness_score = np.mean(result_faithfulness['faithfulness_non_polarity_pure_entailment'])
    coverage_score = np.mean(result_coverage['alignment_wise_pure_positive_scores'])
    f1_score = 2*faithfulness_score*coverage_score/(faithfulness_score+coverage_score) if faithfulness_score+coverage_score>0 else 0.0

    return_dict = {f"{prefix}faithfulness" : round(faithfulness_score*100, 4),
                   f"{prefix}coverage" : round(coverage_score*100, 4),
                   f"{prefix}F1" : round(f1_score*100, 4)}
    return return_dict