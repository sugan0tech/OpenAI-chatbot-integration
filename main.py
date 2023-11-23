import os
import requests
import json
import pinecone
import product
import vector_db
import bot_api
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai_api_key : str = os.getenv("OPENAI_API_KEY")
db_key: str = os.getenv("VECTOR_DB_KEY")


if __name__ == "__main__":
    vector_db.init_db()
    # vector_db.insert_embeddings_to_pinecone_init(product.products)

    print("enter exit to end the bot")
    usr_query = input("Eneter the prompt : ")
    while usr_query != "exit":
        print(bot_api.get_openai_completion(usr_query))
        usr_query = input("Eneter the prompt : ")

