class util:
    def __init__(self):
        pass
    
    @staticmethod
    def tuple2dict( list_tuple , multiplier=1 ):
        return { k:( v * multiplier ) for (k,v) in list_tuple }

    @staticmethod
    def dict2tuple( dict_ , normalising_factor=1):
        return sorted( [ (k,(v/normalising_factor)) for (k,v) in dict_.items() ] , key=lambda x : x[1] , reverse=True )

    @staticmethod
    def list2elastic( positive_terms , negative_terms=[] ):
        term          = lambda term : {"match_phrase":{"content":term}}
        cover_exp_or  = lambda sub_query : {"bool":{"should"   :sub_query}}
        cover_exp_not = lambda sub_query : {"bool":{"must_not" :sub_query}}
        cover_exp_and = lambda sub_query : {"bool":{"must"     :sub_query}}

        final_query   = lambda sub_query : {
                    "from": 0,
                    "size": 10,
                    "track_total_hits": True,
                    "query": sub_query
                }

        pos_neg = []
        pos_neg.append( cover_exp_or( list( map( term ,  positive_terms ) ) ) )
        if ( len( negative_terms ) > 0 ):
            pos_neg.append( cover_exp_and( list( map( term ,  negative_terms ) ) ) )
        
        return final_query( cover_exp_or( pos_neg )  ) 

    @staticmethod
    def ngram(list_tokens, n=2):
        ngram_list = zip(*[list_tokens[i:] for i  in range(n) ])
        return [" ".join(ngram) for ngram in ngram_list  ]
    