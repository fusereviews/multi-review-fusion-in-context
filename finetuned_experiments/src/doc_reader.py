import os
import argparse
import json

class DocReader:
    """
    Reads the docs by id
    """

    def __init__(self, data_json_path: str):
        self.data_json_path = data_json_path
        with open(data_json_path, 'r') as f1:
            self.data_json = json.loads(f1.read())

    def read_review(self, database: str, subset: str, summary_name: str, review_name: str) -> str:
        return [rev for rev in self.data_json[database][subset][summary_name]['reviews_text'] if rev['review_name']==review_name][0]['text']

    def read_summary(self, database: str, subset: str, summary_name: str) -> str:
        return self.data_json[database][subset][summary_name]['summary_text']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--doc_data_dir')
    args, unknown = parser.parse_known_args()
    doc_reader = DocReader(args.doc_data_dir)
    topic = "WSJ910405-0154_d06aa"
    document_file = "WSJ910405-0154"
    doc = doc_reader.read_doc(topic, document_file)
    print(doc)
    summary = doc_reader.read_summary(topic, document_file)
    print(summary)    
