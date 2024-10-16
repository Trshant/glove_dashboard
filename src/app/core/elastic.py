from elasticsearch import Elasticsearch
import types , json

"""
Elasticsearch Mapping Guide: https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping.html
Create Index API: https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-index_.html
Blog on Mapping Basics: https://docs.logz.io/docs/user-guide/data-hub/field-mapping/
"""

settings = {
    "index": {
        "max_result_window": 50000,
        "similarity": {"content_similarity": {"type": "BM25"}},
    },
    "analysis": {
        "normalizer": {
            "my_normalizer": {
                "type": "custom",
                "filter": ["lowercase", "asciifolding"]
            }
        }
    }
}

mapping = {
    "properties": {
        "content": {
            "type": "text",
            "similarity": "content_similarity"
        }
    }
}

class elastic_search:
    def __init__(self, name, es_client=None):
        self.name = name
        if es_client == None:
            self.es_client = Elasticsearch(
                "http://elasticsearch:9200",
                verify_certs=False,
                ssl_show_warn=False,
                basic_auth=("elastic", "JqdGMYPXDGbkFIMLX"),
                http_compress=True
            )
        else:
            self.es_client = es_client
        self.ensure_index_exists()

    def ensure_index_exists(self):
        if not self.es_client.indices.exists(index=self.name):
            self.create_index()

    def create_index(self):
        self.es_client.indices.create(index=self.name, body={"mappings": mapping, "settings": settings })

    def insert_data(self, data, document_id=None, refresh=True):
        """Inserts data into the index.

        Args:
            data (dict): The data to be inserted.
            document_id (str, optional): Optional document ID. If not provided,
                Elasticsearch will generate one.
            refresh (bool, optional): Whether to refresh the index after insertion. Defaults to True.
        """
        if isinstance(data, types.GeneratorType):
            for sentence in data:
                sentence_json = {
                    "content":sentence
                }
                self.es_client.index(index=self.name, body=sentence_json)
        elif isinstance(data, list):
            for sentence in data:
                sentence_json = {
                    "content":sentence
                }
                self.es_client.index(index=self.name, body=sentence_json)
        elif isinstance(data, str):
            sentence_json = {
                "content":data
            }
            self.es_client.index(index=self.name, body=sentence_json)


        if refresh:
            self.es_client.indices.refresh(index=self.name)

    def update_data(self, document_id, data, refresh=True):
        """Updates data in the index.

        Args:
            document_id (str): The ID of the document to update.
            data (dict): The data to update.
            refresh (bool, optional): Whether to refresh the index after update. Defaults to True.
        """
        self.es_client.update(index=self.name, id=document_id, body={"doc": data})
        if refresh:
            self.es_client.indices.refresh(index=self.name)

    def search_index(self, query):
        """

        """
        response = self.es_client.search( index=self.name, body=query )
        return response
