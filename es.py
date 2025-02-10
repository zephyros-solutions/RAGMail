from elasticsearch import Elasticsearch
import requests


class ElSearch:
    ES_URL = "http://localhost:9200/"
    IDX_NAME = 'mails'

    LOW_WTRMK = "2gb"
    HIGH_WTRMK = "1gb"
    FLOOD_ST = "500mb"

    def __init__(self):
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

        # client.indices.delete(index="my_index")
        self.destroy()
        self.client.indices.create(index=self.IDX_NAME)

    def index(self, mails):
        for mail in mails:
            breakpoint()
            self.client.index(
                index=self.IDX_NAME,
                id=mail.CoversationID,
                document={
                    "to": mail.To,
                    "from": mail.From,
                    "subject" : mail.Subject,
                    "content" : mail.Content,
                    "date": mail.Date,
                    "conversationId" : mail.CoversationID
                }
            )

# client.get(index=IDX_NAME, id="my_document_id")

    def search(self):
        self.client.search(index=self.IDX_NAME, query={
            "match": {
                "subject": "dolore"
            }
        })

        breakpoint()
        
        
    def destroy(self):
        # self.client.delete(index=self.IDX_NAME, id="my_document_id")
        self.client.indices.delete(index=self.IDX_NAME)