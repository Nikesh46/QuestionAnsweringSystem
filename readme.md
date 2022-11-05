CS 6320 NLP Project - Question Answering System

Programming tools used:
Python (version: 3.9.5)
Elasticsearch (version: 7.13.4)
NLTK library (version: 3.6.2)
SpaCy library (version: 3.2.0)

Task 1-

    File name - Task1.py

There are two ways to test the script, at document level and at Sentence level.

Command to run -  
 python Task1.py (This will ask for the mode of testing as mentioned below )
Please choose the required method :

1. Features for a Document.
2. Features for a Sentence.

If input "1" is chosen, document level procesing will be chosen and the following text will be displayed.

Enter the file location: (Enter the location of the path. example "./test" )

This will begin the process of extracting NLP Features on the documents present in the inputed directory.

If input "2" or any other character is chosen, sentence level processing will be chosen and the followinf will be displayed.

Enter the sentence: (Enter the sentence. example. "This is a test sentence")

This will output all the NLP features of Task1. (Word Tokens, Word Tokens without Stopwords, POS Tags, POS Tags from WordNet, Stemma Form, Lemma Form, Lemma Form WordNet, Synonyms, Hypernyms, Hyponyms, Meronyms, Holonyms, Dependency Parse Tree, Named Entity Recognition)

Task 2 and Task 3.

Part 1 (Indexing) -
File name - es_index.py

Command to run - python es_index.py

We are using Elasticsearch for storage and Document Retrieval.
Prior to running the index creation file, we will have to first design the Mapping of the Elasticsearch.
Based on the OS the steps of installation varies.
Use the following link for reference:  
 https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html

The Mapping file has been provided as part of the submission ("mapping.json").
Use the following command as create an index with a mapping -

    curl -X PUT "localhost:9200/my-index-000001?pretty" -H 'Content-Type: application/json' -d'
    {"settings": { "analysis": { "analyzer": { "synset": { "tokenizer": "standard","filter": [ "lowercase", "english*stop" ,
    "search_synset": {"tokenizer": "standard", "filter": [ "lowercase", "my_stemmer", "english_stop"]}},
    "filter": {"english_stop": {"type": "stop","stopwords": "\_english*" },
    "synonym" : {"type": "synonym", "format": "wordnet", "synonyms_path": "analysis/wn_s.pl"},
    "my_stemmer": {"type": "stemmer", "language": "english"}}}},
    "mappings" : {"dynamic_templates": [{"text_fields": { "match_mapping_type": "string",
    "mapping": {"type": "text","analyzer" : "synset","copy_to": "unified_field"}}}],
    "properties": {"unified_field": {"type": "text","analyzer" : "synset"
    }}}}'

Run the above Command on terminal after installing elasticsearch.

There's a properties file "config.ini" configured to provide all the important settings to configure elasticsearch and
to configure input file locations.
The script takes in the articles file location and converts them into json file to store them and then use these json files to index data into Elasticsearch.

Details of "config.ini" file

[elastic]
host = localhost (Host where elasticsearch server is runnning)
port = 9200 (Port number of elasticsearch server is runnning)
index_name = suits (index name of elasticsearch is runnning)

[documents]
documents_path = ./articles (Path to input data files )
json_files_path = ./jsons ( path where converted json files are stored for reference.)
task1_location = ./Task1/ (This is the directory which has the Task1 output of all the 30 articles(previously run) and it is also used in Task2.
input_file = testset.txt (File name of the input text file which has a question per line. )
output_file = testset_output.csv ( Ouput file name where the answers to questions are store in the order "Question", "Answer", "Document Id" )

Please make sure the directories provided in the "config.ini" file are present and exact.
also, these document paths are relative of the python file location.

Part 2-
Question and Answering -

The same "config.ini" properties file is used.

File name- qas.py

command to run - python qas.py
This picks up the text file as mentioned in "config.ini" file with the key "input_file" and outputs the csv file with asnwers.
