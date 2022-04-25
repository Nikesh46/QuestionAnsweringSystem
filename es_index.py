from elasticsearch import Elasticsearch, helpers
from configparser import ConfigParser
import glob
import nltk.data
import ssl
import os
import json
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


sentence_tokenizers = nltk.data.load('tokenizers/punkt/english.pickle')

try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

# Read File Content


def read_file_content(file):
    f = open(file, 'r')
    file_content = f.read()
    f.close()
    return file_content

# Configure Elasticsearch


def configure_es(host='localhost', port='9200'):
    config = {'host': host, 'port': port}
    es = Elasticsearch([config, ], timeout=300)
    return es

# Tokenize into Sentences


def sentence_tokenizer(text):
    sentence_tokens = []
    sentence_tokens = sentence_tokenizers.tokenize(text)
    return sentence_tokens


def index_in_es(file_name, json_files_path):

    base = os.path.basename(file_name)

    doc_json = {}
    # Read File Content
    file_content = read_file_content(file_name)

    # Sentence Tokenizer
    sentences = sentence_tokenizer(file_content)

    for index, sentence in enumerate(sentences):
        doc_json[index] = sentences[index]

    output_file_name = os.path.splitext(base)[0]

    with open(json_files_path + '/'+output_file_name+'.json', 'w') as f:
        json.dump(doc_json, f)


def process_data(file_path, json_files_path):
    files = glob.glob(file_path + '/**.txt')
    for file in files:
        index_in_es(file, json_files_path)


def load_json(json_files):
    " Use a generator, no need to load all in memory"
    for filename in json_files:
        with open(filename, 'r') as open_file:
            doc_dict = json.load(open_file)
            doc_dict['_id'] = filename.split('/')[2].split('.')[0]
            yield doc_dict


def es_index(es, index_name, json_files):
    helpers.bulk(es, load_json(json_files), index=index_name, doc_type='_doc')


if __name__ == "__main__":
    print("ES Index")

    config = ConfigParser()
    config.read('config.ini')

    file_path = config.get('documents', 'documents_path')
    json_files_path = config.get('documents', 'json_files_path')

    host = config.get('elastic', 'host')
    port = config.get('elastic', 'port')
    index_name = config.get('elastic', 'index_name')

    es = configure_es(host, port)
    process_data(file_path, json_files_path)
    json_files = glob.glob(json_files_path + '/**.json')
    es_index(es, index_name, json_files)
    print("Elasticsearch Indexing is done")
