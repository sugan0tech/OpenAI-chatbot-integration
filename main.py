import os
import requests
import json
import pinecone
import product
import vector_db
import bot_api
from dotenv import load_dotenv
from openai import OpenAI
import openai
from datasets import load_dataset

load_dotenv()
openai_api_key : str = os.getenv("OPENAI_API_KEY")
db_key: str = os.getenv("VECTOR_DB_KEY")


if __name__ == "__main__":
    index = vector_db.init_db()
    # vector_db.insert_embeddings_to_pinecone_init(product.products)


    print("enter exit to end the bot")
    usr_query = input("Eneter the prompt : ")
    while usr_query != "exit":
        # print(bot_api.get_openai_completion(usr_query))
        res = bot_api.generate_embedding_for_prompt(usr_query)
        embeds = [record['embedding'] for record in res['data']]
        vectors = [ {"values": embeds[0] }]

        matches = index.query(
            vector = embeds[0],
            top_k = 3,
            include_values = True
        )

        print("the matched product is : " + product.products[int(matches["matches"][0]["id"])]["name"])

        usr_query = input("Eneter the prompt : ")

