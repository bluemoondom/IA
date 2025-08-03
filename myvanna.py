# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 17:28:29 2025

@author: dominika
"""

# requires python 3.11
# I added logging but you can remove if you like
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y%m%d %H%M%S')


from vanna.vannadb.vannadb_vector import VannaDB_VectorStore
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
from vanna.ollama import Ollama

class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        Ollama.__init__(self, config=config)
        vs = ChromaDB_VectorStore.__init__(self, config=config)

vn = MyVanna(config={'model': 'mistral'})
vn.connect_to_mssql('Driver=SQL Server;Server=localhost;Trusted_Connection=yes;DATABASE=helios;UID=sa;PWD=Dominique745')
vn.train(
    question="What are the distinct values of ""RadaDokladu"" from the table ""TabDokladyZbozi""?", 
    sql="SELECT RadaDokladu FROM TabDokladyZbozi GROUP BY RadaDokladu"
)
df_information_schema = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")
plan = vn.get_training_plan_generic(df_information_schema)
plan
vn.train(plan=plan)

logging.info('running vannaFlaskApp')
from vanna.flask import VannaFlaskApp
app = VannaFlaskApp(vn, allow_llm_to_see_data=True)
app.run()