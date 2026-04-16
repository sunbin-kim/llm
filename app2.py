import os
from operator import itemgetter

import streamlit as st
from langchain_classic.embeddings.cache import CacheBackedEmbeddings
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.storage import LocalFileStore
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📄",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file_name, file_content, api_key):
    os.makedirs("./.cache/files", exist_ok=True)
    os.makedirs("./.cache/embeddings", exist_ok=True)

    file_path = f"./.cache/files/{file_name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file_name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    return vectorstore.as_retriever()


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for msg in st.session_state["messages"]:
        send_message(msg["message"], msg["role"], save=False)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def load_memory(_):
    return st.session_state["memory"].load_memory_variables({})["history"]


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.

        Context: {context}
        """,
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

# ── Page ──────────────────────────────────────────────────────────────────────

st.title("DocumentGPT")
st.markdown("Upload a `.txt` file in the sidebar, then ask anything about it.")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
    )
    file = st.file_uploader("Upload a .txt file", type=["txt"])
    st.markdown("[GitHub Repository](https://github.com/sunbinkim/DOCUMENT-GPT)")

# ── Session state init ────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
        memory_key="history",
    )

# Reset history when a different file is loaded
if file and st.session_state.get("current_file") != file.name:
    st.session_state["current_file"] = file.name
    st.session_state["messages"] = []
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
        memory_key="history",
    )

# ── Main logic ────────────────────────────────────────────────────────────────

if not openai_api_key:
    st.info("Please enter your OpenAI API key in the sidebar to get started.")
elif not file:
    st.info("Please upload a `.txt` file in the sidebar to get started.")
    st.session_state["messages"] = []
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
        memory_key="history",
    )
else:
    retriever = embed_file(file.name, file.read(), openai_api_key)

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        streaming=True,
        openai_api_key=openai_api_key,
        callbacks=[ChatCallbackHandler()],
    )

    send_message("I'm ready! Ask me anything about your document.", "ai", save=False)
    paint_history()

    user_message = st.chat_input("Ask anything about your file.")
    if user_message:
        send_message(user_message, "human")

        chain = (
            {
                "context": itemgetter("question") | retriever | RunnableLambda(format_docs),
                "question": itemgetter("question"),
                "history": RunnableLambda(load_memory),
            }
            | prompt
            | llm
        )

        with st.chat_message("ai"):
            response = chain.invoke({"question": user_message})

        st.session_state["memory"].save_context(
            {"input": user_message},
            {"output": response.content},
        )
