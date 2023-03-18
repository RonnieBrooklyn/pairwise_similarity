import string
import math
import numpy as np

from numpy.linalg import norm
from typing import Mapping, List, Tuple, Dict
from collections import Counter

from inputs import SAMPLE_DOCS


def convert_doc_to_list(doc:str) -> List:
    """
    Takes raw string of doc and returns list of words minus punctuation and new lines.

    Arguments:
        doc(str)

    Returns
        List of words absent any punctuation.

    *Forgive my use of py3 string.translate()

    """
    return doc.translate(str.maketrans('', '', string.punctuation)).replace('\n', ' ').split(' ')


def convert_doc_words_to_tf(doc:List) -> Mapping[str, int]:
    """
    Takes a single document and returns term frequency for each word in documenbt,
    stripping out punctuation

    term_frequency = occurences / total num words in doc

    Arguments:
        doc(List): list of words in document

    Returns:
        Dict[str, float]: contains term frequency of each word

    Time Cost:
        O(doc_word_count_len) + O(unique_doc_words)

    """
    counts = Counter(doc)

    # bug fix - a null word showing up in text.
    if '' in counts.keys():
        del counts['']

    num_words = sum(counts.values())
    return {word: counts[word] / num_words for word in counts.keys()}


def build_base_doc_db(raw_docs:List[str]) -> Mapping[str, dict]:
    """
    Returns a basic db which contains an id for each doc along with term frequency map for the doc.

    Arguments:
        List[str]: all the strings of docs

    Returns:
        Dict[doc_id] = {
            'tf': Dict[str, float]
            'num_words': int
        }
    """
    doc_db = {}
    for i, one_doc in enumerate(raw_docs):
        doc_list = convert_doc_to_list(one_doc)
        doc_tf = convert_doc_words_to_tf(doc_list)
        doc_db[f'doc_{i}'] = {
            'tf': doc_tf, # TODO: consider CONSTS for dict keys
            'num_words': len(doc_list)
        }
    return doc_db


def normalize_tf(doc_db:Mapping[str, dict]):
    """
    Builds tf for each doc based on a global vocabulary of documents.

    Arguments:
        doc_db[Dict]: collection of tf maps for doc

    Returns:
        doc_db[Dict] with normalized tf dict

    Runtime: O(num_doc*unique_vocab) w O(1) tf lookup.

    """
    vocab = []

    # build vocabulary
    for doc in doc_db:
        vocab.extend(doc_db[doc]['tf'].keys())

    unique_vocab = list(set(vocab))

    for doc in doc_db:
        normalized_tf = {}
        for word in unique_vocab:
            normalized_tf[word] = doc_db[doc]['tf'].get(word, 0)
        doc_db[doc]['tf'] = normalized_tf

    return doc_db, unique_vocab


def compute_word_doc_occurrences(doc_db:Dict, vocab:List) -> Dict:
    """
    Computes number of documents containing each word in unique vocab

    Argument:
        doc_db[Dict]: doc list
        vocab[List]: list of words

    Returns:
        Dict[str, int]: number of docs each word appears in

    Time Cost: O(vocab*num_docs) * O(1) lookup on word

    """
    occurrences = {}
    for word in vocab:
        count = 0
        for doc in doc_db:
            count += math.ceil(doc_db[doc]['tf'].get(word, 0))
        occurrences[word] = count

    return occurrences


def compute_tf_idf(doc_db:Dict, word_occurrences:Dict) -> Dict:
    """
    computes idf and tf_idf for each term in each document

    Argument:
        doc_db[Dict] with each doc containing 'tf' data for doc
        word_occurrences[Dict]

    Returns:
        doc_db[Dict]: {
            'doc_id': {
                'tf': Dict,
                'tf_idf': Dict,
                'idf': Dict,
                'num_words: int
            }
        }

    Time Cost:
        O(num_docs*num_words)*O(1) lookups
    """
    num_docs = len(doc_db.keys())
    for doc in doc_db:
        idf = {}
        tf_idf = {}
        for word in doc_db[doc]['tf']:
            idf[word] = math.log(num_docs / word_occurrences[word])
            tf_idf[word] = idf[word] * doc_db[doc]['tf'][word]

        doc_db[doc]['idf'] = idf
        doc_db[doc]['tf_idf']= tf_idf


    return doc_db


def compute_pairwise_sim(doc1, doc2, unique_vocab):
    """
    Takes tf_idf vectors for each documenat and computes pairwise similarity

    Not sure if allowed to use numpy. Seemed reasonable.

    Args:
        doc1[Dict]: tf_idf map for each word in unique_vocab
        doc2[Dict]: tf_idf map for each word in unique_vocab
        unqiue_vocab: all words in vocab


    """
    # python doesn't promise an order on dicts
    doc1_array = np.array([doc1[val] for val in unique_vocab ])
    doc2_array = np.array([doc2[val] for val in unique_vocab ])
    return np.dot(doc1_array,doc1_array)/(norm(doc1_array)*norm(doc2_array))


def run_pairwise():
    """
    Entry point for tool.
    """
    doc_db = build_base_doc_db(SAMPLE_DOCS)
    doc_db, vocab = normalize_tf(doc_db)
    unique_words_doc_occurrences = compute_word_doc_occurrences(doc_db, vocab)
    doc_db = compute_tf_idf(doc_db, unique_words_doc_occurrences)

    pairwise_sims = {}

    # TODO: fix order independent pairing dupes
    for doc1 in doc_db:
        for doc2 in doc_db:
            pairwise_sims[(doc1, doc2)] = compute_pairwise_sim(
                doc_db[doc1]['tf_idf'],
                doc_db[doc2]['tf_idf'],
                vocab
            )

    print(pairwise_sims)



if __name__ == "__main__":
    run_pairwise()
