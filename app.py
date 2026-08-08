import streamlit as st
from rag_backend import HealthMedBotRAG

st.set_page_config(page_title="HealthMed Bot", page_icon="💊")

st.title("💊 HealthMed Bot")
st.write("Safe OTC Medicine Information Assistant")

# Initialize bot
@st.cache_resource
def load_bot():
    return HealthMedBotRAG()

bot = load_bot()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
query = st.chat_input("Ask about an OTC medicine...")

if query:
    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        response = bot.process_query(query)
        st.write(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
