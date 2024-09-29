import streamlit as st
import pandas as pd
import ollama
import torch
import io
from concurrent.futures import ThreadPoolExecutor

st.title("Data Analysis Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []


# Function to handle file uploads efficiently
def load_data(uploaded_files):
    dataframes = []
    for uploaded_file in uploaded_files:
        # Use smaller chunks to read large files if necessary
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error(f"Unsupported file type: {uploaded_file.name}")
            continue
        dataframes.append(df)
    return dataframes


# Function to handle analysis based on user prompt
def analyze_data(dataframes, prompt):
    # Collect data summaries for each dataframe
    summaries = []
    for i, df in enumerate(dataframes):
        summaries.append(f"File {i + 1} summary:\n{df.describe(include='all')}")

    # Combine all dataframes for analysis
    combined_data = pd.concat(dataframes, axis=0, ignore_index=True)

    # Prompt to ask the model for analysis
    analysis_prompt = f"Perform the following analysis based on the provided data: {prompt}\n\n"
    for summary in summaries:
        analysis_prompt += summary + "\n"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Directly collect the full analysis without streaming to minimize overhead
    model_response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": analysis_prompt}]
    )

    # Extract the full analysis result from the model response
    analysis_result = model_response[0]["message"]["content"]

    return analysis_result, combined_data


# Parallelize the loading of data to make the process faster for multiple files
def parallel_load_data(uploaded_files):
    with ThreadPoolExecutor() as executor:
        dataframes = list(executor.map(load_data, uploaded_files))
    return [df for sublist in dataframes for df in sublist]


# Display chat messages from history on app rerun
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# File upload input
uploaded_files = st.file_uploader("Upload CSV or Excel files", accept_multiple_files=True, type=["csv", "xlsx"])

# Input prompt for user
if prompt := st.chat_input("Enter the analysis prompt here..."):

    # Add user's message to chat history
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # Display user's message in the chat
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process uploaded files
    if uploaded_files:
        # Perform parallelized data loading
        dataframes = load_data(uploaded_files)

        # Perform analysis and get result
        with st.chat_message("assistant"):
            analysis_result, combined_data = analyze_data(dataframes, prompt)
            st.markdown(analysis_result)

        # Allow user to download the combined analyzed data
        buffer = io.BytesIO()
        combined_data.to_excel(buffer, index=False)
        buffer.seek(0)
        st.download_button("Download Analyzed Data", buffer, file_name="analyzed_data.xlsx")

        # Add assistant's response to chat history
        st.session_state["messages"].append({"role": "assistant", "content": analysis_result})
    else:
        st.error("Please upload at least one CSV or Excel file.")
