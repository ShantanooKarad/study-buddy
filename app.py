import os
import random
import time

import chromadb
import streamlit as st
from google import genai
from pypdf import PdfReader

EMBED_MODEL = 'gemini-embedding-001'

st.set_page_config(page_title='Study Buddy', page_icon='📚')
st.title('📚 Study Buddy: ask questions about your notes')


def get_setting(name, default=None):
    # Look in environment variables first, then in Streamlit secrets.
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets[name]
    except Exception:
        return default


API_KEY = get_setting('GEMINI_API_KEY')
CHAT_MODEL = get_setting('CHAT_MODEL', 'gemini-3.8-flash')

if not API_KEY:
    st.error('GEMINI_API_KEY is missing. Add it to .streamlit/secrets.toml (local) or to Secrets (Streamlit Cloud).')
    st.stop()

client = genai.Client(api_key=API_KEY)


@st.cache_resource
def get_collection():
    # In-memory vector database, created once and reused.
    database = chromadb.Client()
    return database.get_or_create_collection(name='notes', metadata={'hnsw:space': 'cosine'})


collection = get_collection()

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'indexed' not in st.session_state:
    st.session_state.indexed = []
if 'failed' not in st.session_state:
    st.session_state.failed = []


def split_text(text, size=1000, overlap=200):
    # Cut text into 1000-character chunks that overlap by 200 characters.
    chunks = []
    start = 0
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(piece)
        start += size - overlap
    return chunks


def embed_batch(batch):
    for attempt in range(5):
        try:
            result = client.models.embed_content(model=EMBED_MODEL, contents=batch)
            return [item.values for item in result.embeddings]
        except Exception as error:
            # 429 means the free-tier limit was hit: wait and try again.
            if '429' not in str(error) or attempt == 4:
                raise
            time.sleep(20)


def embed(texts):
    vectors = []
    for i in range(0, len(texts), 20):
        vectors.extend(embed_batch(texts[i:i + 20]))
        time.sleep(2)
    return vectors


def index_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    ids, docs, metas = [], [], []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ''
        for n, chunk in enumerate(split_text(text)):
            ids.append(f'{uploaded_file.name}-p{page_number}-c{n}')
            docs.append(chunk)
            metas.append({'source': uploaded_file.name, 'page': page_number})
    if not docs:
        return 0
    collection.upsert(ids=ids, documents=docs, embeddings=embed(docs), metadatas=metas)
    return len(docs)


def ask_gemini(prompt):
    for attempt in range(4):
        try:
            response = client.models.generate_content(model=CHAT_MODEL, contents=prompt)
            return response.text
        except Exception as error:
            # 503 = Google is busy, 429 = free-tier limit: wait and try again.
            busy = '503' in str(error) or '429' in str(error)
            if not busy or attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))


# ---------- Sidebar: upload notes and make a quiz ----------
with st.sidebar:
    st.header('1. Upload your notes')
    files = st.file_uploader('PDF files', type='pdf', accept_multiple_files=True)
    for f in files or []:
        if f.name in st.session_state.indexed or f.name in st.session_state.failed:
            continue
        with st.spinner('Reading ' + f.name + ' ...'):
            try:
                count = index_pdf(f)
            except Exception as error:
                st.session_state.failed.append(f.name)
                st.error('Could not index ' + f.name + ': ' + str(error))
                continue
        if count == 0:
            st.session_state.failed.append(f.name)
            st.warning('No text found in ' + f.name + '. It may be a scanned PDF.')
        else:
            st.session_state.indexed.append(f.name)
    if st.session_state.indexed:
        st.success('Indexed: ' + ', '.join(st.session_state.indexed))
    st.header('2. Practice')
    make_quiz = st.button('Generate a 5-question quiz')

if make_quiz:
    all_chunks = collection.get()['documents']
    if not all_chunks:
        st.sidebar.warning('Upload a PDF first.')
    else:
        sample = random.sample(all_chunks, min(8, len(all_chunks)))
        material = '\n\n'.join(sample)
        quiz_prompt = f'''Write 5 multiple-choice questions (4 options each) based ONLY on the study material below.
After the questions, give an answer key.

STUDY MATERIAL:
{material}'''
        with st.spinner('Writing your quiz ...'):
            try:
                st.session_state.quiz = ask_gemini(quiz_prompt)
            except Exception as error:
                st.session_state.quiz = 'Could not make a quiz: ' + str(error)

if st.session_state.get('quiz'):
    with st.expander('Your quiz', expanded=True):
        st.markdown(st.session_state.quiz)

# ---------- Chat ----------
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

question = st.chat_input('Ask a question about your notes')
if question:
    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message('user'):
        st.markdown(question)

    with st.chat_message('assistant'):
        sources = ''
        if collection.count() == 0:
            answer = 'Please upload a PDF first (left sidebar).'
        else:
            with st.spinner('Thinking ...'):
                try:
                    question_vector = embed_batch([question])[0]
                    hits = collection.query(query_embeddings=[question_vector], n_results=4)
                    parts = []
                    for i, (chunk, meta) in enumerate(zip(hits['documents'][0], hits['metadatas'][0]), start=1):
                        header = '[Source {}: {}, page {}]'.format(i, meta['source'], meta['page'])
                        parts.append(header + '\n' + chunk)
                    sources = '\n\n'.join(parts)
                    prompt = f'''You are a helpful study assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say: I could not find this in your notes.
Mention the source numbers you used, like [Source 1].

CONTEXT:
{sources}

QUESTION: {question}'''
                    answer = ask_gemini(prompt)
                except Exception as error:
                    answer = 'Something went wrong: ' + str(error)
        st.markdown(answer)
        if sources:
            with st.expander('Show sources'):
                st.text(sources)
    st.session_state.messages.append({'role': 'assistant', 'content': answer})
