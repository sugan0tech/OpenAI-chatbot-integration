import pinecone
import os
import bot_api
import product
from dotenv import load_dotenv
load_dotenv()

pinecone_key = os.getenv("VECTOR_DB_KEY")


index_name = "chat-bot-index"

def init_db():

    pinecone.init(pinecone_key, environment="gcp-starter")
    print(pinecone.list_indexes())

    if index_name in pinecone.list_indexes():
        print("index present")
    else:
        pinecone.create_index(index_name, dimension=1536, metric="euclidean")
        pinecone.describe_index(index_name)
        insert_embeddings_to_pinecone_init(product.products)

    return pinecone.Index(index_name)



# a working sample 
def insert_embeddings(embeddings):
    index = init_db()

    index.upsert(
    vectors=[
        {"id": "A", "values": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]},
        {"id": "B", "values": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]},
        {"id": "C", "values": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]},
        {"id": "D", "values": [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]},
        {"id": "E", "values": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]}
      ]
    )

def insert_embeddings_to_pinecone_init(product_data):
    index = pinecone.Index(index_name)
    for i, product in enumerate(product_data):

        res = bot_api.generate_embedding(product)
        embeds = [record['embedding'] for record in res['data']]
        vectors = [{"id": str(i + 1) , "values": embeds[0]}]
        print(embeds)
        index.upsert(vectors=vectors)

