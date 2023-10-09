from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings.openai import OpenAIEmbeddings
from data_loader import draw_column_separator, extract_table_data
from src import openai_config

# use gpt-3.5-turbo model and create embeddings
llm = ChatOpenAI(
    temperature=0, model_name="gpt-3.5-turbo", openai_api_key=openai_config.api_key
)
embeddings = OpenAIEmbeddings(openai_api_key=openai_config.api_key)


def initialize_retrieval_qa(docs, kwargs):
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        verbose=False,
        chain_type="stuff",
        chain_type_kwargs=kwargs,
    )


def prompt_engineering():
    # add information from the dictionary as context
    dictionary_path = "resources/dictionary.png"
    processed_dictionary = draw_column_separator(dictionary_path)
    df = extract_table_data(processed_dictionary)

    context = df["Concatenated"]

    # This can be changed as per the users needs
    prompt_template = """f"Only answer questions using information from the table. You can engage in small talk. If you don't know 
        say I dont know\nContext:{context}\nQuestion:{question}\nAnswer:"
        """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    return {"prompt": PROMPT}
