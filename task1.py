import ssl
import sys
from itertools import chain
import os
import glob
import errno
import json
import spacy

import nltk
import nltk.data
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('corpora/wordnet')
    from nltk.corpus import wordnet as wn

except:
    nltk.download('wordnet')

sentence_tokenizers = nltk.data.load('tokenizers/punkt/english.pickle')

nlp = spacy.load('en_core_web_sm')

all_stopwords = nlp.Defaults.stop_words
all_stopwords.remove('not')

lemmatizer = WordNetLemmatizer()


# Read File Content


def read_file_content(file):
    f = open(file, 'r')
    file_content = f.read()
    f.close()
    return file_content


# Tokenize Articles into Sentences


def sentence_tokenizer(text):
    sentence_tokens = []
    sentence_tokens = sentence_tokenizers.tokenize(text)

    return sentence_tokens

# Words Extracted from a Sentence


def word_tokenizer(sentence):
    return word_tokenize(sentence)

# POS Tags for Words using NLTK POS Taggers


def pos_taggers(word_tokens):
    return nltk.pos_tag(word_tokens)

# Tagging POS Tags through the Wordnet Corpus


def wordNet_pos_tagger(nltk_tag):

    if nltk_tag.startswith('J'):
        return wn.ADJ

    elif nltk_tag.startswith('V'):
        return wn.VERB

    elif nltk_tag.startswith('N'):
        return wn.NOUN

    elif nltk_tag.startswith('R'):
        return wn.ADV

    else:
        return None

# Word Stemmatization NLTK Based - to find of Stemmas of the Words


def word_stemmatization(words):
    stemmatize_word = {}
    ps = PorterStemmer()
    for word in words:
        stemmatize_word[word] = ps.stem(word)
    return stemmatize_word

# Dependency Parsing on the Sentence - to find the dependency in words, tags and the tree structure of the sentence


def dependency_parsing(sentence):
    dependency_parsed_tree = []
    doc = nlp(sentence)
    sent = list(doc.sents)
    for s in sent:
        rootOfSentence = s.root.text
    for token in doc:
        dependency_parsed_tree.append(
            [token.dep_, token.head.text, token.text])
    return dependency_parsed_tree

# Named Entity Recognition Function to extract entities based on patterns


def named_entity_recognition(sentence):
    ner = {}
    doc = nlp(sentence)
    for X in doc.ents:
        ner[str(X)] = X.label_

    return ner

# Wordnet Features - Synonymns, Hypernymns, Hyponymns, Meronymns, Holonymns


def wordnet_features(words):

    synonymns = {}
    hypernyms = {}
    hyponyms = {}
    meronyms = {}
    holonyms = {}

    # Looping through Words
    for word in words:
        temp_synonymns = []
        temp_hypernyms = []
        temp_hyponyms = []
        temp_meronyms = []
        temp_holonyms = []

        # Synsets for the Word (WordNet)
        for i, j in enumerate(wn.synsets(word)):

            # Adding the synonymns to the List
            temp_synonymns.extend(wn.synset(j.name()).lemma_names())

            # Adding the hypernymns to the List
            temp_hypernyms.extend(
                list(chain(*[l.lemma_names() for l in j.hypernyms()])))

            # Adding the hyponymns to the List
            temp_hyponyms.extend(
                list(chain(*[l.lemma_names() for l in j.hyponyms()])))

            # Adding the meronymns to the List
            temp_meronyms.extend(
                list(chain(*[l.lemma_names() for l in j.part_meronyms()])))

            # Adding the holonymns to the List
            temp_holonyms.extend(
                list(chain(*[l.lemma_names() for l in j.part_holonyms()])))

        # Adding to the Dictionary
        synonymns[word] = temp_synonymns
        hypernyms[word] = temp_hypernyms
        hyponyms[word] = temp_hyponyms
        meronyms[word] = temp_meronyms
        holonyms[word] = temp_holonyms

    return synonymns, hypernyms, hyponyms, meronyms, holonyms

# Lemmatization of Words


def lemmatization(word_tokens):

    lemmas = {}

    for word in word_tokens:
        lemmas[word] = lemmatizer.lemmatize(word)

    return lemmas

# Lemmatization of Words - with regard to the Wordnet tgged words


def lemmatization_wordnet(wordnet_tagged):

    lemmas_wordnet = {}

    for word, tag in wordnet_tagged:
        if tag is None:
            lemmas_wordnet[word] = []
        else:
            lemmas_wordnet[word] = lemmatizer.lemmatize(word, tag)

    return lemmas_wordnet

# Creation of the Feature Pipeline for NLP


def NLP_Feature_Pipeline(sentence):

    # Word Tokens from the Sentence without Stop Words
    word_tokens = [word for word in word_tokenizer(
        sentence) if not word in all_stopwords]

    # Word Tokens from the Sentence with Stop Words
    word_tokens_sw = [word for word in word_tokenizer(
        sentence)]

    # Dependency Parsing Tree
    d_parse = dependency_parsing(sentence)

    # Wordnet Features - Synonymns, Hypernymns, Hyponymns, Meronymns, Holonymns
    syn, hyper, hypo, mero, holo = wordnet_features(word_tokens)

    # Stemmatization
    stemmas = word_stemmatization(word_tokens)

    # Attach POS Tags to the Word List
    pos_tagged = pos_taggers(word_tokens)

    # Attaching the Wordnet POS Tags to the NLTK POS tags
    wordnet_tagged = list(
        map(lambda x: (x[0], wordNet_pos_tagger(x[1])), pos_tagged))

    # Initialization of the Lemmas for a WordNet Tagged Sentence
    lemmas_wordnet = lemmatization_wordnet(wordnet_tagged)

    # Initialization of the Lemmas for a Sentence
    lemmas = lemmatization(word_tokens)

    # Spacy - Named Entity Recognition for Sentence
    ner = named_entity_recognition(sentence)

    return word_tokens, word_tokens_sw, pos_tagged, wordnet_tagged, stemmas, lemmas, lemmas_wordnet, syn, hyper, hypo, mero, holo, d_parse, ner


# ------------------------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------- Task 1 - NLP Features from Input Text File --------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------------------------ #
def Features_Extraction(file_path):

    base = os.path.basename(file_path)
    # path = file_path.split("/")[:]
    path = file_path.split("/")[1]

    sentence_json = {}
    words_json = {}
    words_sw_json = {}
    pos_tag_json = {}
    wordnet_tagged_json = {}
    stemmas_json = {}
    lemmas_json = {}
    lemmas_wordnet_json = {}
    synonymns_json = {}
    hypernyms_json = {}
    hyponyms_json = {}
    meronyms_json = {}
    holonyms_json = {}
    dependency_parse_tree_json = {}
    ners_json = {}

    print("-----------------------------------------------------------------------------------------------------------")
    print('\nStarting Task 1 - Feature Extraction from Text File - '+base+'\n')

    # Read text File
    file_content = read_file_content(file_path)

    # Sentence Tokenizer
    sentences = sentence_tokenizer(file_content)

    # For each sentence with the index, fetch all the requried features.
    for index, sentence in enumerate(sentences):

        word_tokens, word_tokens_sw, pos_tagged, wordnet_tagged, stemma_json, lemmas, lemmas_wordnet, synonymn_json, hypernym_json, hyponym_json, meronym_json, holonym_json, dependency_parse_tree, ner_json = NLP_Feature_Pipeline(
            sentence)

        sentence_json[index] = sentence
        words_json[index] = word_tokens
        words_sw_json[index] = word_tokens_sw
        pos_tag_json[index] = pos_tagged
        wordnet_tagged_json[index] = wordnet_tagged
        stemmas_json[index] = stemma_json
        lemmas_json[index] = lemmas
        lemmas_wordnet_json[index] = lemmas_wordnet
        synonymns_json[index] = synonymn_json
        hypernyms_json[index] = hypernym_json
        hyponyms_json[index] = hyponym_json
        meronyms_json[index] = meronym_json
        holonyms_json[index] = holonym_json
        dependency_parse_tree_json[index] = dependency_parse_tree
        ners_json[index] = ner_json

    output_file_name = os.path.splitext(base)[0]

    try:
        os.makedirs(path + "/" + output_file_name)

    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    with open(path + "/" + output_file_name+'/'+output_file_name+'_Sentence_Tokens.json', 'w') as filehandle:
        json.dump(sentence_json, filehandle)
    with open(path + "/" + output_file_name+'/'+output_file_name+'_Word_Tokens.json', 'w') as filehandle:
        json.dump(words_json, filehandle)
    with open(path + "/" + output_file_name+'/'+output_file_name+'_Word_Tokens_Stop_Words.json', 'w') as filehandle:
        json.dump(words_sw_json, filehandle)
    with open(path + "/" + output_file_name+'/'+output_file_name+'_POS_Tags.json', 'w') as filehandle:
        json.dump(pos_tag_json, filehandle)
    with open(path + "/" + output_file_name+'/'+output_file_name+'_Stemmas.json', 'w') as filehandle:
        json.dump(stemmas_json, filehandle)
    with open(path + "/" + output_file_name+'/'+output_file_name+'_Lemmas.json', 'w') as filehandle:
        json.dump(lemmas_json, filehandle)
    with open(path + "/" + output_file_name+'/'+output_file_name+'_Lemmas_WordNet.json', 'w') as filehandle:
        json.dump(lemmas_wordnet_json, filehandle)
    with open(path + "/" + output_file_name+'/'+output_file_name+'_Dependency_Parse_Tree.json', 'w') as filehandle:
        json.dump(dependency_parse_tree_json, filehandle)
    with open(path + "/" + output_file_name+'/'+output_file_name+'_synonyms.json', 'w') as filehandle:
        json.dump(synonymns_json, filehandle)
    with open(path + "/" + output_file_name+'/'+output_file_name+'_hypernymns.json', 'w') as filehandle:
        json.dump(hypernyms_json, filehandle)
    with open(path + "/" + output_file_name+'/'+output_file_name+'_hyponymns.json', 'w') as filehandle:
        json.dump(hyponyms_json, filehandle)
    with open(path + "/" + output_file_name+'/'+output_file_name+'_meronymns.json', 'w') as filehandle:
        json.dump(meronyms_json, filehandle)
    with open(path + "/" + output_file_name+'/'+output_file_name+'_holonymns.json', 'w') as filehandle:
        json.dump(holonyms_json, filehandle)
    with open(path + "/" + output_file_name+'/'+output_file_name+'_NERs.json', 'w') as filehandle:
        json.dump(ners_json, filehandle)

    return sentences, words_json, pos_tag_json, wordnet_tagged_json, stemmas_json, lemmas_json, lemmas_wordnet_json, synonymns_json, hypernyms_json, hyponyms_json, meronyms_json, holonyms_json, dependency_parse_tree_json, ners_json


def process_data(file_path):
    files = glob.glob(file_path + '/**.txt')
    for file in files:
        Features_Extraction(file)


if __name__ == "__main__":
    print('QA System')

    choice = input(
        "Please choose the required method : \n1. Features for a Document. \n2. Features for a Sentence.\n")

    if choice == "1":
        file_path = input("Enter the file location:\n")

        process_data(file_path)
    else:
        sentence = input("Enter the sentence:\n")
        # sentence = sys.argv[0]

        word_tokens, words_sw_json, pos_tagged, wordnet_tagged, stemma_json, lemmas, lemmas_wordnet, synonymn_json, hypernym_json, hyponym_json, meronym_json, holonym_json, dependency_parse_tree, ner_json = NLP_Feature_Pipeline(
            sentence)
        print("Sentence: ", sentence, "\n")
        print("Word Tokens : ", words_sw_json, "\n")
        print("Word Tokens without stopwords :", word_tokens, "\n")
        print("POS Tags ", pos_tagged, "\n")
        print("WordNet Tags: ", wordnet_tagged, "\n")
        print("Stemma Form ", stemma_json, "\n")
        print("Lemma form: ", lemmas, "\n")
        print("Lemma form WordNet: ", lemmas_wordnet, "\n")
        print("Synonyms: ", synonymn_json, "\n")
        print("Hypernyms: ", hypernym_json, "\n")
        print("Hyponym: ", hyponym_json, "\n")
        print("Meronym: ", meronym_json, "\n")
        print("Holonym: ", holonym_json, "\n")
        print("Dependency Parse Tree: ", dependency_parse_tree, "\n")
        print("Named Entity Recognition: ", ner_json, "\n")
