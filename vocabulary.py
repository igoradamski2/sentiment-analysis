    
class Vocabulary:

    @staticmethod
    def getUniGrams(train_set):
        
        # Lowercase all letters
        vocabulary = set()
        for documents in train_set:
            for words in documents:
                vocabulary.add(words)

        return vocabulary

    @staticmethod
    def getBiGrams(train_set):

        vocabulary = set()
        for documents in train_set:
            for bigram in zip(documents, documents[1:]):
                vocabulary.add(bigram[0] + " " + bigram[1])

        return vocabulary

    @staticmethod
    def getUnigramsByDocument(document):

        assert type(document) == list

        wordDict = {}
        for words in document:
            if words not in wordDict.keys():
                wordDict[words] = 1
            else:
                wordDict[words] += 1

        return wordDict

    @staticmethod
    def getBigramsByDocument(document):

        assert type(document) == list

        wordDict = {}
        for bigram in zip(document, document[1:]):
            mybigram = bigram[0] + " " + bigram[1]
            if mybigram not in wordDict.keys():
                wordDict[mybigram] = 1
            else:
                wordDict[mybigram] += 1

        return wordDict

    @staticmethod
    def getVocabularyByDocument(document, grams):

        if 'uni' in grams:
            full_vocab = Vocabulary.getUnigramsByDocument(document)
        else:
            full_vocab = {}

        if 'bi' in grams:
            full_vocab2 = Vocabulary.getBigramsByDocument(document)
        else:
            full_vocab2 = {}
        
        full_vocab.update(full_vocab2)

        return full_vocab

    @staticmethod
    def getFullDict(document_set, vocabulary, grams):

        # We will go through the documents and 
        # get dictonaries for each. Then we will
        # add the occurences to our master dictionary
        # which contains all of vocabulary

        occurrences = {word:0 for word in vocabulary}
        
        for document in document_set:
            # Get vocab for that document
            doc_vocab = Vocabulary.getVocabularyByDocument(document, grams)

            for words in doc_vocab.keys():

                occurrences[words] += doc_vocab[words]

        return occurrences



















