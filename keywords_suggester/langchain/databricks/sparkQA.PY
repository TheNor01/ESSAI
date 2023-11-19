from langchain.document_loaders import PySparkDataFrameLoader

from pyspark.sql import SparkSession
from langchain.agents import create_spark_dataframe_agent

from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.utilities.spark_sql import SparkSQL

spark = SparkSession.builder.getOrCreate()


"""
teams_df = spark.read.csv("resources/mlb_teams_2012.csv", header=True)
loader = PySparkDataFrameLoader(spark, teams_df, page_content_column="Team")
documents = loader.load()
"""

df = spark.read.csv("resources/mlb_teams_2012.csv", header=True, inferSchema=True)
df.show()


print("COUNT")
print(df.count())


callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm_llama= LlamaCpp(
    # You can pass in the URL to a GGML model to download it automatically
    model_path="resources/codellama-7b-instruct.Q5_K_M.gguf",
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    temperature=0,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager, 
    verbose=True, # Verbose is required to pass to the callback manager
)


agent = create_spark_dataframe_agent(llm=llm_llama, df=df, verbose=True)


query = "how many rows are there? "
query = query + " using tool python_repl_ast"

agent.run(query)

query = "give me the average payroll"
query = query + " using tool python_repl_ast"

agent.run(query)

agent.run('give me pyspark code to store my df into delta format')