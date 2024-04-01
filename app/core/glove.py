from gensim.models import Word2Vec
from app.libs.utils import util
from pathlib import Path
import os, types, re
from collections import Counter

OUTPUT_PATH="./case_data/"

class glove:
    def __init__(self, case_name):
        """
        Constructor method (initializer).
        Initialize instance variables here.
        """
        self.caseName = case_name
        self.model =  self.load_model()

    def load_model(self):
        """
        This method loads the glove model from the file.
        """
        filepath = OUTPUT_PATH+self.caseName+"/model.v3.pkl"
        if Path(filepath).is_file() :
            model = Word2Vec.load(filepath)
            return model
        else :
            return False

    def create_model(self, list_list_token):
        """
        This method created the model from the one sentence you will give it.
        """
        if len( list_list_token) == 1:
            self.model = Word2Vec(sentences=list_list_token )

        self.model = Word2Vec(sentences=list_list_token, min_count=10, workers=4)
    
    def save_model(self):
        """
        This method saves the model to a local filepath
        """
        path = OUTPUT_PATH + self.caseName
        os.makedirs(path, exist_ok=True)
        self.model.save(path+"/model.v3.pkl")
    
    def tokeniser(self, text):
        return re.findall(r"\w+", text.lower())


    def add_sentences(self, sentences):
        """
        This method is the main 
        """
        if isinstance(sentences, types.GeneratorType):
            if self.model == None:
                    self.create_model( [ self.tokeniser( next(sentences) ) ]  )
                    return True
            count = 0
            c_sentences = []
            for sentence in sentences:
                c_sentences.append( self.tokeniser( sentence ) )
                count = count + 1
                if count % 10000 == 0 :
                    print(count)
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
