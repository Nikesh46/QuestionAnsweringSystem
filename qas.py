from elasticsearch import Elasticsearch
import json
import spacy
from nltk.tokenize import word_tokenize
import numpy as np
from nltk.corpus import wordnet as wn
from configparser import ConfigParser
from nltk.stem import WordNetLemmatizer
import pandas as pd


lemmatizer = WordNetLemmatizer()
nlp = spacy.load('en_core_web_sm')
all_stopwords = nlp.Defaults.stop_words
# all_stopwords.remove('not')


# Build Elastic Search query


def return_query_result(q, es, index):
    result = es.search(
        index=index,
        body={
            "from": 0,
            "size": 10,
            "query": {
                "query_string": {
                    "query": q,
                    "fields": ['unified_field']
                }
            },
            "highlight": {
                "require_field_match": "false",
                "order": "score",
                "highlight_query": {
                    "query_string": {
                        "query": q,
                        "fields": ['unified_field']

                    }
                },
                "fields": {
                    "*": {}
                }
            },
            "_source": "false"
        }
    )
    return result

# Configure Elasticsearch host


def configure_es(host='localhost', port='9200'):
    config = {'host': host, 'port': port}
    es = Elasticsearch([config, ], timeout=300)
    return es

# Word Tokenizer


def word_tokenizer(sentence):
    return word_tokenize(sentence)

# Query Processor


def query_processer(q):
    query_filter = {"who", "when", "what", "?"}
    processed_query = " ".join([word for word in word_tokenize(
        q) if not word.lower() in query_filter])
    # processed_query = " ".join(q.split(" ")[1:])[:-1]
    return processed_query

# Ranking Algorithm


def cos_sim(doc_field_ids, q):
    cos_sim = []
    new_doc_field = []
    qt_pq = q.split(" ")
    qt, pq = qt_pq[0].lower(), " ".join(qt_pq[1:])

    query_tokens = {token.lower() for token in word_tokenize(pq
                                                             ) if not token in all_stopwords}

###############################################################################################
    # Query Root Word
    ques = nlp(q)
    ques_list = list(ques.sents)
    for s in ques_list:
        query_root = s.root.text

    # Query Root Word's Synonyms
    for i, j in enumerate(wn.synsets(query_root)):
        query_synonyms = {w for w in wn.synset(
            j.name()).lemma_names() if w not in all_stopwords}

    # Query Type Rule Based
    query_entities = set()
    if "who" == qt:
        query_entities.add("ORG")
        query_entities.add("PERSON")
    elif "when" in qt:
        query_entities.add('DATE')
        query_entities.add('TIME')


################################################################################################
    # Answer Processing
    for doc, field in doc_field_ids:
        # print(doc, field)
        filename = data_location + doc+'/'+doc + "_Sentence_Tokens.json"
        with open(filename, 'r') as open_file:
            doc_dict = json.load(open_file)
        sentence = doc_dict[field]

#################################################################################################################
        # Root of Sentence  Boost:
        dependency_parsed_tree = []
        ans_head_set = set()
        ans_doc = nlp(sentence)
        sent = list(ans_doc.sents)
        for s in sent:
            ans_root = s.root.text
        for token in ans_doc:
            dependency_parsed_tree.append(
                [token.dep_, token.head.text, token.text])
            ans_head_set.add(token.head.text)

        # NER Entity Boost
        ner_ans = set()
        ner_ans_type = set()

        nerq = nlp(sentence)
        for X in nerq.ents:
            ner_ans.add(str(X))
            ner_ans_type.add(X.label_)

################################################################################################################
        # Boosting and Scoring
        answer_tokens = {
            token.lower() for token in word_tokenize(sentence) if not token in all_stopwords}
        rv = answer_tokens.union(query_tokens)
        answer_vector = []
        question_vector = []
        for w in rv:
            if w in query_tokens:
                question_vector.append(1)
            else:
                question_vector.append(0)
            if w in answer_tokens:
                if w in ner_ans:
                    answer_vector.append(20)
                elif w in query_tokens:
                    answer_vector.append(10)
                else:
                    answer_vector.append(1)
            else:
                answer_vector.append(0)

        # Root of Query and Answer matching
        if ans_root == query_root:
            question_vector.append(1)
            answer_vector.append(10)

        elif lemmatizer.lemmatize(ans_root) == lemmatizer.lemmatize(query_root):
            question_vector.append(1)
            answer_vector.append(10)

        # Root of Query and answer heads Match
        if query_root in ans_head_set:
            question_vector.append(1)
            answer_vector.append(5)

        # Query Root's Synonym match with answer heads
        for query_head_syn in query_synonyms:
            if query_head_syn in ans_head_set:
                question_vector.append(1)
                answer_vector.append(3)

        # Rule based Question Type Matching
        if query_entities:
            for qe in query_entities:
                if qe in ner_ans_type:
                    question_vector.append(1)
                    answer_vector.append(7)
                    break

##############################################################################################
        count = 0
        for index in range(len(rv)):
            count += answer_vector[index] * question_vector[index]
        current_score = count / \
            float((sum(answer_vector) * sum(question_vector))**0.5)

        new_doc_field.append(((doc, field), sentence))
        cos_sim.append(current_score)

    return new_doc_field, np.array(cos_sim)


if __name__ == "__main__":
    print('ES Query')

    config = ConfigParser()
    config.read('config.ini')

    host = config.get('elastic', 'host')
    port = config.get('elastic', 'port')
    index_name = config.get('elastic', 'index_name')
    data_location = config.get('documents', 'task1_location')
    input_file = config.get('documents', 'input_file')
    output_file = config.get('documents', 'output_file')

    es = configure_es(host, port)

    testfile = open(input_file, 'r')
    queries = testfile.read().splitlines()

    topn = 5

    count = 0

    df = pd.DataFrame(columns=["Query", "Answer", "Document Id"])
    for index, q in enumerate(queries):
        processed_query = query_processer(q)
        print("--------------------------------------------------------------------------------------------------")
        print("Query -->", q)
        print("--------------------------------------------------------------------------------------------------")
        result = return_query_result(processed_query, es, index_name)

        doc_field_ids = []
        for hit in result['hits']['hits']:
            for field, val in hit['highlight'].items():
                if field != "unified_field":
                    doc_field_ids.append((hit['_id'], field))
        new_doc_field_ids, cossim = cos_sim(
            doc_field_ids, q)

        sorted_cossim = np.argsort(cossim)

        cossim_filter = []
        top_n = 0
        for index, val in enumerate(sorted_cossim[::-1]):
            if cossim[val] > 0:
                cossim_filter.append((new_doc_field_ids[val], cossim[val]))
                top_n += 1

            if top_n == topn:
                break
        doc_id = cossim_filter[0][0][0][0]
        answer = cossim_filter[0][0][1]
        df = df.append(pd.DataFrame([[q, answer, doc_id]], columns=[
                       "Query", "Answer", "Document Id"]), ignore_index=True)
        # for ndfs, score in cossim_filter:
        #     print(ndfs, score)
    df.to_csv(output_file)
