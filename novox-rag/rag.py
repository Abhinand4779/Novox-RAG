from fastapi import FastAPI
import requests
from bs4 import BeautifulSoup
import redis
import uuid
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams,Distance,PointStruct
from langchain_ollama import ChatOllama

URLS =[    
    "https://www.python.org/",
    "https://react.dev/",
    "https://www.djangoproject.com/",
    "https://fastapi.tiangolo.com/",
    "http://novoxedtechllp.com/",
]

COLLECTION_NAME = "institution_rag"


app = FastAPI()

redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
print("FastAPI Loaded")
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Loading reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Reranker Loaded Successfully")

print("Loading LLM...")
llm = ChatOllama(model="llama3")

print("Loaded the Chatollama")

print("All models loaded!")

qdrant = QdrantClient(path="./qdrant_data")

if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
vector_size = 384

def fetch_url(url):
    try:
        response = requests.get(url, timeout=20)

        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        text = soup.get_text(separator=" ").strip()

        return Document(page_content=text, metadata={"source": url})
  
    except Exception as e:
        print("Fetch error:", e)
        return None

def load_docs():
    docs = []
    for url in URLS:
        print(f"Fetching: {url}")
        doc = fetch_url(url)
        if doc:
            docs.append(doc)
    return docs

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
    )
    return splitter.split_documents(docs)
 
def store_chunks(chunks):
    texts = [c.page_content for c in chunks]
    embeddings = embedder.encode(texts)
    points = []

    for chunk, vector in zip(chunks, embeddings):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload={
                    "text": chunk.page_content,
                    "source": chunk.metadata.get("source", ""),
                },
            )
        )

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Stored {len(points)} vectors")


@app.get("/")
def home():
    return {"message": "Industry RAG running 🚀"}

@app.get("/ingest")
def ingest():
    try:
        docs = load_docs()
        chunks = split_docs(docs)
        store_chunks(chunks)

        return {"message": f"Ingested {len(chunks)} chunks"}

    except Exception as event:
        return {"error": str(event)}




@app.get("/query")
def query_rag(q: str):
    try:
        print(f"Query: {q}")

        cached = redis_client.get(q)
        if cached:
            return {"response": cached}

        query_vector = embedder.encode(q).tolist()

        results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=8,
        ).points


    
        if not results:
            return {"response": "No relevant data found."}

        
        pairs = [(q, r.payload.get("text", "")) for r in results]
        scores = reranker.predict(pairs)

        ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)

        if ranked[0][1] < 0.3:
            return {"response": "No relevant answer found."}


        top_results = [r for r, _ in ranked[:3]]

        context = "\n\n".join([
            r.payload.get("text", "") for r in top_results
        ])



        prompt = f"""
You are an expert AI assistant.

Answer the question in a clear, detailed, and well-structured paragraph.

Instructions:
- Use simple language
- Explain clearly (like teaching a beginner)
- Answer should be at least 5-6 lines
- Do NOT repeat the question
- Do NOT mention "based on context"

If the answer is not present in the context, say:
"I don't know."

Context:
{context}

Question:
{q}

Detailed Answer:
"""


        response = llm.invoke(prompt)
        answer = response.content.strip()


        if not answer or len(answer) < 20:
            return {"response": "No detailed answer found."}

        sources = list(set([
            r.payload.get("source", "") for r in top_results
        ]))

        final_answer = f"{answer}\n\nSources:\n" + "\n".join(sources)

        redis_client.set(q, final_answer)

        return {"response": final_answer}
    

    except Exception as event:
        return {"error": str(event)}



