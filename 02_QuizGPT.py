import streamlit as st
import json
from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_classic.embeddings.cache import CacheBackedEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.storage import LocalFileStore
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.output_parsers import BaseOutputParser


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()


st.set_page_config(
    page_title="QuizGPT",
    page_icon="?",
)

st.title("QuizGPT")

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-5-nano",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """ 
                You are a helpful assistant that is role playing as a teacher.

                Based ONLY on the following context make 10 quesitons to test the user's knowledge about the text.

                Each question should have 4 answers, three of them must be incorrect and one should be correct.

                Use (o) to signal the correct answer.

                Question examples:

                Question: What is the color of the ocean?
                Answers: Red | Yellow | Green | Blue(o)

                Question: What is the capital of Georgia?
                Answers: Baku | Tbilisi(o) | Manila | Beirut

                Question: When was Avatar released?
                Answers: 2007 | 2001 | 2009(o) | 1998

                Question: Who was Julius Caesar?
                Answers: A Roman Emperor(o) | Painter | Actor | Model

                Your turn!

                Context: {context}
                """,
        )
    ]
)

questions_chain = {"context": format_docs} | questions_prompt | llm

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """ 
            You are a powerful formatting algorithms.

            You format exam questions into JSON format.
            Answers with (o) are the correct ones.

            Example Input:

            Question: What is the color of the ocean?
            Answers: Red | Yellow | Green | Blue(o) 

            Question: What is the capital or Georgia?
            Answers: Baku | Tbilisi(o) | Manila | Beirut

            Question: When was Avatar released?
            Answers: 2007 | 2001 | 2009(o) | 1998 

            Quesiton: Who was Julius Caesar?
            Answers: A Roman Emperor(o) | Painter | Actor | Model

            Example Output:

            ```json # starting point
            {{ "questions": [
                {{
                    "question": "What is the color of the ocean?",
                    "answers": [
                        {{
                            "answer": "Red",
                            "correct": false
                        }},
                        {{
                            "answer": "Yellow",
                            "correct": false
                        }},
                        {{
                            "answer": "Green",
                            "correct": false
                        }},
                        {{
                            "answer": "Blue",
                            "correct": true
                        }},
                        ]
            }},
                    {{
                    "question": "What is the capital of Georgia?",
                    "answers": [
                        {{
                            "answer": "Baku",
                            "correct": false
                        }},
                        {{
                            "answewr": "Tbilisi",
                            "correct": true
                        }},
                        {{
                            "answer": Manila",
                            "correct": false
                        }},
                        {{
                            "answer": "Beirut",
                            "correct": false
                        }},
                        
                        
                        ]
                    
                    }},
                    {{
                "question": "When was Avatar released?",
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }},
                ]
            }},
            {{
                "question": "Who was Julius Caesar?",
                "answers": [
                        {{
                            "answer": "A Roman Emperor",
                            "correct": true
                        }},
                        {{
                            "answer": "Painter",
                            "correct": false
                        }},
                        {{
                            "answer": "Actor",
                            "correct": false
                        }},
                        {{
                            "answer": "Model",
                            "correct": false
                        }},
                ]
            }}
            ]
            }}
            ``` # complete without json without extra conversation

            Your turn!

            Questions: {context}
            """,
        )
    ]
)

formatting_chain = formatting_prompt | llm


@st.cache_resource(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    st.write(file_content, file_path)
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    # load the document
    loader = TextLoader(file_path)

    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_resource(show_spinner="Making quiz...")
def run_quiz_chain(
    _docs, topic
):  # if the parameter 'topic' chagnes, the function will run again
    chain = {"context": questions_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)


@st.cache_resource(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.invoke(term)
    return docs


with st.sidebar:
    docs = None
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx, .txt or .pdf file",
            type=["pdf", "txt", "docs"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            # retriever = WikipediaRetriever(top_k_results=5)
            # docs = retriever.get_relevant_documents(topic)
            # with st.status("Searching Wikipedia"):
            #    docs = retriever.invoke(topic)
            # st.write(docs)
            docs = wiki_search(topic)

# if we don't have documents
if not docs:
    st.markdown(
        """ 
    Welcome to QuizGPT.

    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.

    Get started by uploading a file or searching on Wikipedia in the sidebar.
        """
    )
# if we do have documents
else:
    # st.write(docs)
    # questions_response = questions_chain.invoke(docs)
    # st.write(questions_response.content)
    # formatting_response = formatting_chain.invoke({
    #    "context": questions_response.content
    # })
    response = run_quiz_chain(docs, topic if topic else file.name)
    #st.write(response)
    with st.form("questions_form"):
        for question in response["questions"]:
            st.write(question["question"])
            value = st.radio(
                "Select an option.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
            )
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct!")
            elif value is not None:
                st.error("Wrong!")
        button = st.form_submit_button()
