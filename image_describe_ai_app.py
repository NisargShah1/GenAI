import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from PIL import Image
from io import BytesIO
import io
import base64
from langchain_core.messages import HumanMessage

#  Streamlit UI
st.title("📸 AI-Powered Image Describer with QnA")

if "image_description" not in st.session_state:
    st.session_state.image_description = None

# Live streaming handler
class StreamHandler(BaseCallbackHandler):
  def __init__(self, container, initial_text=""):
    self.container = container
    self.text = initial_text

  def on_llm_new_token(self, token: str, **kwargs) -> None:
    self.text += token
    self.container.markdown(self.text)

# Load Gemini LLM
gemini_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    max_tokens=1024,
    convert_system_message_to_human=True,
)

# Define prompt structure
SYSTEM_PROMPT = "You are a friendly AI image describer. Help users with their queries."

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# LLM Chain
llm_chain = prompt | gemini_model

# Store conversation history in Streamlit session state
streamlit_msg_history = StreamlitChatMessageHistory()

# Conversation chain with memory
conversation_chain = RunnableWithMessageHistory(
    llm_chain,
    lambda session_id: streamlit_msg_history,  # Memory store
    input_messages_key="input",
    history_messages_key="history",
)
    
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # Convert image to PNG format
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# Step 1: Upload Image
uploaded_image = st.file_uploader("Upload an image for AI analysis", type=["jpg", "png", "jpeg"])

if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Convert image to bytes
    image = Image.open(uploaded_image)
    image = image.resize((512, 512))
    image_data = image_to_base64(image)

    streamlit_msg_history.add_user_message("describe image")

    # Step 2: AI Describes the Image
    with st.spinner("Analyzing image..."):
        message = HumanMessage(
          content=[
            {"type": "text", "text": "describe image"},
            {
              "type": "image_url",
              "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            },
          ]
        )
        response = gemini_model.invoke([message])
        st.session_state.image_description = response.content
        streamlit_msg_history.add_ai_message(response.content)

# Use cached result
if st.session_state.image_description:
    st.markdown(f"**AI Description:** {st.session_state.image_description}")


for msg in streamlit_msg_history.messages:
    st.chat_message(msg.type).write(msg.content)

# User input box
if user_input := st.chat_input("Ask a question about the image..."):
    st.chat_message("human").write(user_input)

    # Stream response
    with st.chat_message("ai"):
        stream_handler = StreamHandler(st.empty())

        # Run LLM with user input
        response = conversation_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "user_session"}, "callbacks": [stream_handler]},
        )
        # Show final response to ensure complete output is displayed
        st.markdown(response.content)
