from pairwise import (
    convert_doc_to_list,
    convert_doc_words_to_tf,
    build_base_doc_db,
    normalize_tf,
    compute_word_doc_occurrences,
    compute_tf_idf,
    compute_pairwise_sim,
)

TEST_DOC_2 = "we love our dog."
TEST_DOC = "the dog ran with another dog."
TEST_DOC_LIST = ["the", "dog", "ran", "with", "another", "dog"]

TEST_DOC_WORDS_TF = {
    "the": 1 / 6,
    "dog": 2 / 6,
    "ran": 1 / 6,
    "with": 1 / 6,
    "another": 1 / 6,
}

TEST_BASE_DOC_DB = {"doc_0": {"tf": TEST_DOC_WORDS_TF, "num_words": 6}}

TEST_UNIQUE_VOCAB = ["the", "dog", "ran", "with", "another"]


def test_convert_doc_to_list():
    assert convert_doc_to_list(TEST_DOC) == TEST_DOC_LIST


def test_convert_doc_words_to_tf():
    assert convert_doc_words_to_tf(TEST_DOC_LIST) == TEST_DOC_WORDS_TF


def test_build_base_doc_db():
    assert build_base_doc_db([TEST_DOC]) == TEST_BASE_DOC_DB


def test_normalize_tf():
    doc_db = build_base_doc_db([TEST_DOC, TEST_DOC_2])
    actual_doc_db, actual_vocab = normalize_tf(doc_db)
    assert len(actual_vocab) == 8

    # make sure vocabs match between docs
    assert actual_doc_db["doc_0"]["tf"].keys() == actual_doc_db["doc_1"]["tf"].keys()


def test_compute_word_occurrences():
    doc_db = build_base_doc_db([TEST_DOC, TEST_DOC_2])
    actual_doc_db, actual_vocab = normalize_tf(doc_db)

    actual_word_occurrences = compute_word_doc_occurrences(actual_doc_db, actual_vocab)

    assert actual_word_occurrences["dog"] == 2
    assert actual_word_occurrences["love"] == 1


def test_compute_tf_idf():
    doc_db = build_base_doc_db([TEST_DOC, TEST_DOC_2])
    actual_doc_db, actual_vocab = normalize_tf(doc_db)
    actual_word_occurrences = compute_word_doc_occurrences(actual_doc_db, actual_vocab)

    actual_doc_db_tfidf = compute_tf_idf(actual_doc_db, actual_word_occurrences)

    # same idf for same word that appears in both docs
    assert actual_doc_db_tfidf["doc_1"]["idf"]["dog"] == 0
    assert actual_doc_db_tfidf["doc_0"]["idf"]["dog"] == 0

    # words not in docs have 0 tfidf
    assert actual_doc_db_tfidf["doc_1"]["tf_idf"]["ran"] == 0
    assert actual_doc_db_tfidf["doc_0"]["tf_idf"]["we"] == 0


def test_pairwise_sim():
    doc_db = build_base_doc_db([TEST_DOC, TEST_DOC_2])
    actual_doc_db, actual_vocab = normalize_tf(doc_db)
    actual_word_occurrences = compute_word_doc_occurrences(actual_doc_db, actual_vocab)

    actual_doc_db_tfidf = compute_tf_idf(actual_doc_db, actual_word_occurrences)

    # maximum pairwise sim for tifidf of same doc
    assert (
        compute_pairwise_sim(
            actual_doc_db_tfidf["doc_0"]["tf_idf"],
            actual_doc_db_tfidf["doc_0"]["tf_idf"],
            actual_vocab,
        )
        == 1
    )
