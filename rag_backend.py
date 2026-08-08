# rag_backend.py - FINAL CLEAN VERSION (NO FORCE FORMAT)

import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

load_dotenv()

# ----------------------------
# PATH
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

# ----------------------------
# MINIMAL SAFETY (Doctor suggestion)
# ----------------------------
BLOCKED_DRUGS = ["aspirin", "loperamide"]

# ----------------------------
# STRONG PROMPT (THIS CONTROLS FORMAT)
# ----------------------------
SYSTEM_PROMPT = """
You are a healthcare assistant providing OTC medicine suggestions.

Follow these STRICT instructions:

1. Always suggest medicines relevant to the symptoms.
2. Do NOT repeat same medicines for all queries.
3. NEVER suggest aspirin or loperamide.
4. NEVER say "unknown".
5. Keep answer clean and structured.

RESPONSE FORMAT (VERY IMPORTANT):

OTC MEDICINE RECOMMENDATION
1. <Medicine Name>
2. <Medicine Name>

SAFETY ADVICE
1. <Advice>
2. <Advice>

Rules:
- Medicine section should contain ONLY medicine names
- Safety advice should NOT contain medicines
- Do not mix sections

Context:
{context}

User Query:
{question}
"""

# ----------------------------
# CLASS
# ----------------------------
class HealthMedBotRAG:

    def __init__(self):

        print("🔄 Initializing HealthMedBot...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        self.db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=self.embeddings
        )

        self.llm = Ollama(
            model="phi3:mini",
            temperature=0.3,
            num_ctx=1024,
            num_predict=150
        )

        self.prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

        print("✅ Clean RAG Bot Ready")

    # ----------------------------
    # 🔥 QUERY EXPANSION
    # ----------------------------
    def expand_query(self, query):
        return f"""
        Patient symptoms: {query}
        What OTC medicines are typically used for this condition?
        """

    # ----------------------------
    # 🔥 RETRIEVAL FIX (IMPORTANT)
    # ----------------------------
    def retrieve_context(self, query):

        expanded_query = self.expand_query(query)

        docs = self.db.similarity_search(expanded_query, k=3)

        print(f"📚 Retrieved {len(docs)} docs")

        context = "\n\n".join(doc.page_content for doc in docs)

        return context[:1200]

    # ----------------------------
    # SAFETY FILTER (ONLY BLOCK)
    # ----------------------------
    def safety_filter(self, response):

        response_lower = response.lower()

        if any(drug in response_lower for drug in BLOCKED_DRUGS):
            return (
                "⚠️ Some medicines may not be safe for self-medication.\n"
                "Please consult a healthcare professional."
            )

        return response

    # ----------------------------
    # MAIN FUNCTION
    # ----------------------------
    def process_query(self, query: str):

        print("\n🔎 Query:", query)

        # Get context
        context = self.retrieve_context(query)

        # Prompt
        final_prompt = self.prompt.format(
            context=context,
            question=query
        )

        # LLM
        response = self.llm.invoke(final_prompt)

        # Safety
        response = self.safety_filter(response)

        return response


# ----------------------------
# TEST
# ----------------------------
if __name__ == "__main__":

    bot = HealthMedBotRAG()

    print(bot.process_query("I have cough and headache"))
    print(bot.process_query("I have redness and swelling in wound"))
    print(bot.process_query("I feel nausea and weakness"))