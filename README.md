# pairwise_similarity
Computes pairwise similarity of term-frequency inverse document frequency vectors of sample text. 

# Committment
Spent 2-3 hours on this while watching March Madness on Friday, March 17 (took breaks when games were compelling!) 

#References
https://towardsdatascience.com/text-vectorization-term-frequency-inverse-document-frequency-tfidf-5a3f9604da6d
https://www.geeksforgeeks.org/how-to-calculate-cosine-similarity-in-python/
https://stackoverflow.com/questions/34293875/how-to-remove-punctuation-marks-from-a-string-in-python-3-x-using-translate


text - https://constitutioncenter.org/the-constitution/full-text


# To Improve
+Under load of many docs might benefit from some optimization
+Prints pairiwser similarity with some dupes



# To Run
PYTHONPATH=. python pairwise.py

# To Test
Ran out of time, but code is written in a modular fashion to allow unit testing of each fn. 
One macro unit test is to be sure pairwise of one doc with itself is 1. That is tested in output of fun, 

