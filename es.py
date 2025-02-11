from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from nltk.corpus import stopwords

import requests
from tqdm import tqdm
import time

class ElSearch:
    ES_URL = "http://localhost:9200/"
    IDX_PRFX = 'mails'

    LOW_WTRMK = "2gb"
    HIGH_WTRMK = "1gb"
    FLOOD_ST = "500mb"

    def __init__(self, mailsId):
        self.client = Elasticsearch(
            self.ES_URL,  # Elasticsearch endpoint
            api_key="",
        )

        requests.put(f"{self.ES_URL}_cluster/settings", json = {
        "transient": {
            "cluster.routing.allocation.disk.watermark.low": self.LOW_WTRMK,
            "cluster.routing.allocation.disk.watermark.high": self.HIGH_WTRMK,
            "cluster.routing.allocation.disk.watermark.flood_stage": self.FLOOD_ST,
            "cluster.info.update.interval": "5m"
        }
        })
        self.index = f'{self.IDX_PRFX}_{mailsId.lower()}'
        
        # self.client.indices.delete(index=self.IDX_PRFX)
        self.destroy()
        # mappings = {}
        settings = {
            "analysis": {
                "filter": {
                    "italian_elision": {
                        "type": "elision",
                        "articles": [
                                "c", "l", "all", "dall", "dell",
                                "nell", "sull", "coll", "pell",
                                "gl", "agl", "dagl", "degl", "negl",
                                "sugl", "un", "m", "t", "s", "v", "d"
                        ],
                        "articles_case": True
                    },
                    "italian_stop": {
                        "type":       "stop",
                        "stopwords":  stopwords.words('italian')
                    },
                    "italian_keywords": {
                        "type":       "keyword_marker",
                        "keywords":   [] 
                    },
                    "italian_stemmer": {
                        "type":       "stemmer",
                        "language":   "italian"
                    },
                    "italian_snowball": {
                        "type": "snowball",
                        "language": "Italian"
                    }
                },
                "analyzer": {
                    "italian_analyzer": {
                    "tokenizer": "standard",
                    "char_filter": [
                        "html_strip"
                    ],
                    "filter": [
                        "italian_elision",
                        "lowercase",
                        "italian_stop",
                        "italian_keywords",
                        "italian_snowball"
                        ]
                    }
                },
            }
        }
        mappings = {
            'dynamic': False,
            'properties': {
                'date': {
                    'type': 'date'
                },
                'from': {
                    'type': 'text',
                    'fields': {
                        'keyword': {'type': 'keyword', 'ignore_above': 256}
                    }
                },
                'to': {
                    'type': 'text',
                    'fields': {
                        'keyword': {'type': 'keyword', 'ignore_above': 256}
                    }
                },
                'subject': {
                    'type': 'text',
                    "analyzer": "italian_analyzer", 
                    'fields': {
                        'keyword': {'type': 'keyword', 'ignore_above': 256}
                    }
                },
                'content': {
                    'type': 'text',
                    "analyzer": "italian_analyzer", 
                    # 'fields': {
                    #     'keyword': {'type': 'keyword', 'ignore_above': 256}
                    # }
                },
                'conversationId': {
                    'type': 'long'
                },
            }
        }

        result = self.client.indices.create(index=self.index, mappings=mappings, settings=settings)
        if not result['acknowledged']:
            # we should wait instead as operation might be successful but late
            raise Exception(f'Creating index {self.index} failed')
        # breakpoint()


    def index_mails(self, mails):
        items = []
        for mail in tqdm(mails,desc="Indexing emails"):
            # breakpoint()
            source = {
                    "to": mail.To,
                    "from": mail.From,
                    "subject" : mail.Subject,
                    "content" : mail.Content,
                    "date": mail.Date,
                    "conversationId" : mail.CoversationID
                }
            item = {
                '_index': self.index,
                '_id' : mail.CoversationID,
                '_source' : source
            }
            items.append(item)

            # result = self.client.index(
            #     index=self.index,
            #     id=mail.CoversationID,
            #     document={
            #         "to": mail.To,
            #         "from": mail.From,
            #         "subject" : mail.Subject,
            #         "content" : mail.Content,
            #         "date": mail.Date,
            #         "conversationId" : mail.CoversationID
            #     }
            # )
        result = bulk(self.client, items)
        if result[0] != len(mails):
            raise Exception(f'Errors during bulk indexing: {result}')
        # breakpoint()
        properties = self.client.indices.get_mapping(index=self.index)[f'{self.index}']['mappings']['properties']
        print(f'The mapping for {self.index} is {properties}')
        # breakpoint()

    def search(self, prompt):
        mail_ids = []

        while True:
            result = self.client.search(index=self.index, query={
                "multi_match": {
                    "query": prompt,
                    "fields": ["subject^2", "content"]
                }
            })
            print(result)
            if result['took'] == 0 and len(result['hits']['hits']) == 0:
                time.sleep(1)
            else:
                break
        if not ('hits' in result and 'hits' in result['hits']):
            breakpoint()    
        
        for hit in result['hits']['hits']:
            # breakpoint()
            mail_ids.append(hit['_id'])
        
        return mail_ids

        
        
        
    def destroy(self):
        # self.client.delete(index=self.index, id="my_document_id")
        if self.client.indices.exists(index=self.index):
            # breakpoint()
            print(f"Deleting index {self.index}")
            self.client.indices.delete(index=self.index)



# {
#   "settings": {
#     "analysis": {
#       "analyzer": {
#         "my_analyzer": {
#           "tokenizer": "keyword",
#           "char_filter": [
#             "html_strip"
#           ]
#         }
#       }
#     }
#   }
# }

