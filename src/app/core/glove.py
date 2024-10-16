from gensim.models import Word2Vec
from app.libs.utils import util
from pathlib import Path
import os, types, re
from collections import Counter

import sys
from functools import lru_cache



OUTPUT_PATH="./case_data/"

class glove:
    def __init__(self, case_name):
        """
        Constructor method (initializer).
        Initialize instance variables here.
        """
        sys.setrecursionlimit(15000)
        self.caseName = case_name
        self.model =  self.load_model()

    def load_model(self):
        """
        This method loads the glove model from the file.
        """
        filepath = OUTPUT_PATH+self.caseName+"/model.v3.pkl"
        if Path(filepath).is_file() :
            print( "found model file!" )
            model = Word2Vec.load(filepath)
            return model
        else :
            return None

    def create_model(self, list_list_token):
        """
        This method created the model from the one sentence you will give it.
        """
        if len( list_list_token) > 1:
            self.model = Word2Vec(sentences=list_list_token, min_count=10, workers=4)
            return True
        self.model = Word2Vec(sentences=list_list_token )

    def save_model(self):
        """
        This method saves the model to a local filepath
        """
        path = OUTPUT_PATH + self.caseName
        os.makedirs(path, exist_ok=True)
        self.model.save(path+"/model.v3.pkl")
    
    def tokenise(self, tokens , point=0):
        if point >= len(tokens):
            return []
        if point == 0:
            return [ tokens[point] ] + self.tokenise(tokens, (point + 1))
        if point == 1:
            return [ tokens[point-1]+" "+tokens[point] , tokens[point] ] + self.tokenise(tokens, (point + 1))
        if point > 1:
            return [ tokens[point-2]+" "+tokens[point-1]+" "+tokens[point] , tokens[point-1]+" "+tokens[point] , tokens[point] ] + self.tokenise(tokens, (point + 1))


    
    def tokeniser(self, text):
        tok =  re.findall(r"\w+", text.lower())
        print(len(tok))
        ngrammed_tok = self.tokenise(tok )
        return ngrammed_tok


    def add_sentences(self, sentences):
        """
        This method is the main 
        """
        if isinstance(sentences, types.GeneratorType):
            count = 0
            c_sentences = []
            for sentence in sentences:
                c_sentences.append( self.tokeniser( sentence ) )
                count = count + 1
                if count % 10000 == 0 :
                    ## print(count)
                    if self.model == None:
                        self.create_model( c_sentences  )
                        continue
                    self.model.build_vocab(c_sentences, update=True)
                    self.model.train(c_sentences, total_examples=self.model.corpus_count, epochs=self.model.epochs)
                    c_sentences = []
            return True
        
        print("not a generator, using normal way to build model")

        if self.model == None:
            self.create_model(sentences)
            return True    

    def get_similar_words(self,words,cnt,neg_words=[], filter_negatives=True):
        wordCounter = Counter()
        print( "model : ", self.model )
        for word in words:
            embeddings = self.model.wv.most_similar(positive=[word], topn=cnt )
            wordCounter.update( util.tuple2dict( embeddings , 1) )
        for word in neg_words:
            embeddings = self.model.wv.most_similar(positive=[word], topn=cnt )
            wordCounter.update( util.tuple2dict( embeddings , (-1.25)) )
        list_tuple = util.dict2tuple(wordCounter, (len(words) + len(neg_words) ))
        if filter_negatives == True:
            filter_fn = lambda x: False if x[1] < 0 else True
            return list( filter( filter_fn , list_tuple ) )[:cnt]
        return list_tuple


# Usage example:
if __name__ == "__main__":

    case_name = "IMDB"    
    words = ["good","great","superb","excellent"]
    count=15
    neg_words = [] ## ["australia","scandinavia","branch"]
    filter_negatives = True

    obj = glove("IMDB")
    dd = obj.get_similar_words(words,count,neg_words,filter_negatives)
    print(dd)
