import os
import time
import json
import datetime
import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd

# Load environment variables
load_dotenv()

EXCEL_FILE = "script_outputs.xlsx"

def save_to_excel(topic: str, output: str):
    row = {"Topic": topic, "Script": output}
    if os.path.exists(EXCEL_FILE):
        df = pd.read_excel(EXCEL_FILE)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_excel(EXCEL_FILE, index=False)

def setup_gemini_llm():
    """Initialize Gemini LLM for script-writer agent"""
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7,
        max_retries=10,
        retry_min_seconds=5,
        retry_max_seconds=30,
        streaming=True
    )

def build_script_prompt(topic: str) -> str:
    """Builds a prompt for the script-writer agent based on the topic."""
    return f"""
You are Script-Writer, an agent that creates simple, engaging scripts for any topic. 
Given a topic, break it down into the following sections, answering each in a clear, friendly, and informative way. Use bullet points or short paragraphs. If you don't know, say so simply.

Topic: {topic}

I'll approach it, keeping it simple and engaging:
1. Who:
   - Who are the people or groups involved in this topic?
   - Who is most affected or benefits from it?
2. What:
   - What is this topic about in simple terms?
   - What are some surprising or interesting facts about it?
3. When:
   - When did this topic become important?
   - When were key changes or events related to it?
4. Where:
   - Where does this topic matter the most?
   - Where have important things about it happened?
"""

def script_writer_agent(user_input: str) -> str:
    """Handles input like 'topic- <subject>' and returns a structured script."""
    if user_input.lower().startswith("topic-"):
        topic = user_input[6:].strip()
        if not topic:
            return "Please provide a topic after 'topic-'."
        llm = setup_gemini_llm()
        prompt = build_script_prompt(topic)
        try:
            response = llm.invoke(prompt)
            if isinstance(response, dict) and 'content' in response:
                output = response['content']
            else:
                output = str(response)
            save_to_excel(topic, output)
            return output
        except Exception as e:
            return f"Error generating script: {str(e)}"
    else:
        return "Please use the input style: 'topic- <subject>' (e.g., 'topic- indian economy')."

def main():
    """Main function to run the script-writer agent."""
    print("📝 Script-Writer Agent is ready!")
    print("Type 'quit' to exit\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        if not user_input:
            continue
        print("Script-Writer thinking...")
        output = script_writer_agent(user_input)
        print(f"\nScript-Writer:\n{output}\n")

if __name__ == "__main__":
    main()
