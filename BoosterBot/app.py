import streamlit as st
import ollama
import time
import pandas as pd
import os
import re

# Function to stream data word by word
def stream_data(text, delay: float = 0.02):
    for word in text.split():
        yield word + " "
        time.sleep(delay)

# Function to read CSV file
def read_csv_file(file_path):
    return pd.read_csv(file_path)

# Function to filter relevant entries based on the keywords
def filter_relevant_entries(dataframe, keywords):
    # Escape each keyword to make them safe for regex
    escaped_keywords = [re.escape(keyword) for keyword in keywords]
    query = '|'.join(escaped_keywords)
    filtered_df = dataframe[dataframe.apply(lambda row: row.astype(str).str.contains(query, case=False, na=False).any(), axis=1)]
    
    # Score and rank entries based on relevance
    def relevance_score(row):
        score = 0
        for keyword in keywords:
            if keyword.lower() in str(row).lower():
                score += 1
        return score

    filtered_df['RelevanceScore'] = filtered_df.apply(relevance_score, axis=1)
    ranked_df = filtered_df.sort_values(by='RelevanceScore', ascending=False)
    return ranked_df.head(3)  # Limit to top 3 entries

# Function to extract and summarize relevant information and citations
def extract_and_summarize(filtered_df):
    summary_list = []
    citations = []

    for index, row in filtered_df.iterrows():
        entry = f"Title: {row['Title']}\nDate: {row['DateIssue']}\n\n{row['MainText']}\n\nDiscussion: {row['Discussion']}\n"
        summary_list.append(entry)
        if 'Citation' in row:
            citations.append(row['Citation'])

    summarized_content = "\n\n".join(summary_list)  # Use all selected entries for summarization
    return summarized_content, citations

# Path to the uploaded CSV file
csv_file_path = "./Data/MMWRSet1.csv"

# Read the CSV file
dataframe = read_csv_file(csv_file_path)

# Initialize session state for chat history if not already done
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Input for Prompt
prompt = st.chat_input("Ask about MMWR Content")

if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Step 1: Extract keywords from the user's query using the model
    with st.spinner("Extracting keywords..."):
        keyword_result = ollama.chat(model="mistral:latest", messages=[{
            "role": "user",
            "content": f"Extract the main keywords from the following query: {prompt}"
        }])
        keywords = keyword_result["message"]["content"].split()

    # Step 2: Filter relevant entries from the dataframe using the extracted keywords
    relevant_entries = filter_relevant_entries(dataframe, keywords)

    # Step 3: Extract and summarize relevant information and citations
    summarized_content, citations = extract_and_summarize(relevant_entries)

    # Step 4: Generate response using the summarized content
    full_prompt = f"""
    You are an expert in MMWR content. You are a friendly, helpful assistant like one would find in a library. Use the following summarized data from our archives to answer the user's query accurately. At the end of your response, include any relevant citations from the archives:

    {summarized_content}

    User's question: {prompt}
    """

    with st.spinner("Generating response..."):
        result = ollama.chat(model="mistral:latest", messages=[{
            "role": "user",
            "content": full_prompt
        }])
        response = result["message"]["content"]

        # Add relevant citations to the response
        if citations:
            relevant_citations = "\n\nRelevant citations:\n" + "\n".join(citations[:3])  # Limit to top 3 citations
            full_response = response + relevant_citations
        else:
            full_response = response

        # Add model response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="./Images/userprofile.jpg"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant", avatar="./Images/boosterbotprofile.jpg"):
            st.write_stream(stream_data(message["content"]))
