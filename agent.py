import os
import time
import json
import datetime
import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain import hub
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

def setup_gemini_llm():
    """Initialize Gemini LLM with rate limiting"""
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7,
        max_retries=10,
        retry_min_seconds=5,
        retry_max_seconds=30,
        streaming=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )

def create_sample_tools():
    """Create sample tools for the agent"""
    def calculator(expression: str) -> str:
        """Calculate mathematical expressions"""
        try:
            result = eval(expression)
            return f"The result is: {result}"
        except Exception as e:
            return f"Error calculating: {str(e)}"
    
    def text_length(text: str) -> str:
        """Count characters in text"""
        return f"Text length: {len(text)} characters"
    
    def get_date_time(timezone: str = None) -> str:
        """Get current date and time, optionally for a specific timezone"""
        now = datetime.datetime.now()
        if timezone:
            try:
                # Simple timezone handling for common timezones
                offsets = {
                    "utc": 0, "gmt": 0,
                    "est": -5, "edt": -4,
                    "cst": -6, "cdt": -5,
                    "mst": -7, "mdt": -6,
                    "pst": -8, "pdt": -7,
                }
                timezone = timezone.lower()
                if timezone in offsets:
                    offset = datetime.timedelta(hours=offsets[timezone])
                    now = datetime.datetime.utcnow() + offset
                    return f"Current date and time in {timezone.upper()}: {now.strftime('%Y-%m-%d %H:%M:%S')}"
                else:
                    return f"Timezone '{timezone}' not supported. Try UTC, GMT, EST, CST, MST, PST, etc."
            except Exception as e:
                return f"Error processing timezone: {str(e)}"
        return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
    
    def get_weather(location: str) -> str:
        """Get current weather for a location"""
        try:
            # Using a free weather API that doesn't require authentication
            url = f"https://wttr.in/{location}?format=j1"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                current = data["current_condition"][0]
                temp_c = current["temp_C"]
                temp_f = current["temp_F"]
                desc = current["weatherDesc"][0]["value"]
                humidity = current["humidity"]
                
                return f"Weather in {location}: {desc}, {temp_c}°C ({temp_f}°F), Humidity: {humidity}%"
            else:
                return f"Error fetching weather: HTTP {response.status_code}"
        except Exception as e:
            return f"Error getting weather: {str(e)}"
    
    def search_web(query: str) -> str:
        """Search the web for information using a built-in knowledge base"""
        # Common knowledge base for frequently asked questions
        knowledge_base = {
            "capital of india": "The capital of India is New Delhi.",
            "oldest tree": "Methuselah, a Great Basin bristlecone pine in California, is among the oldest known living trees at over 4,800 years old.",
            "top monuments in india": "The top 5 monuments to visit in India are:\n1. Taj Mahal (Agra) - A white marble mausoleum built by Emperor Shah Jahan in memory of his wife Mumtaz Mahal.\n2. Red Fort (Delhi) - A historic fort that served as the main residence of the Mughal Emperors.\n3. Qutub Minar (Delhi) - The tallest brick minaret in the world and a UNESCO World Heritage Site.\n4. Gateway of India (Mumbai) - An arch monument built during the 20th century to commemorate the landing of King George V and Queen Mary.\n5. Hawa Mahal (Jaipur) - A palace known for its unique five-story exterior with 953 small windows called jharokhas.",
            "taj mahal": "The Taj Mahal is a white marble mausoleum located in Agra, India. It was built by Emperor Shah Jahan between 1631 and 1648 in memory of his favorite wife, Mumtaz Mahal. It's considered one of the most beautiful buildings in the world and a masterpiece of Mughal architecture, combining elements from Persian, Islamic, and Indian architectural styles. The Taj Mahal is a UNESCO World Heritage Site and one of the New Seven Wonders of the World.",
            "red fort": "The Red Fort is a historic fort in Old Delhi, India, that served as the main residence of the Mughal Emperors. Built in 1639 by Emperor Shah Jahan, it's known for its massive red sandstone walls. Every year on India's Independence Day (August 15), the Prime Minister hoists the national flag at the Red Fort and delivers a nationally broadcast speech.",
            "qutub minar": "Qutub Minar is a 73-meter tall minaret in Delhi, India. Construction began in 1193 under Qutb al-Din Aibak and was completed by his successor. It's the tallest brick minaret in the world and a UNESCO World Heritage Site, known for its intricate carvings and verses from the Quran.",
            "gateway of india": "The Gateway of India is an arch monument in Mumbai, built during the 20th century to commemorate the landing of King George V and Queen Mary at Apollo Bunder in 1911. It's a popular tourist attraction and a symbol of Mumbai, located on the waterfront overlooking the Arabian Sea.",
            "hawa mahal": "Hawa Mahal (Palace of Winds) is a palace in Jaipur, India, built in 1799 by Maharaja Sawai Pratap Singh. It has a unique five-story exterior resembling a honeycomb with 953 small windows called jharokhas. These windows allowed royal ladies to observe street festivals while remaining unseen, as they had to follow strict 'purdah' (face covering).",
            "monuments in india": "India is home to numerous historical monuments including the Taj Mahal (Agra), Red Fort (Delhi), Qutub Minar (Delhi), Gateway of India (Mumbai), Hawa Mahal (Jaipur), Ajanta and Ellora Caves (Maharashtra), Konark Sun Temple (Odisha), Khajuraho Temples (Madhya Pradesh), Hampi Ruins (Karnataka), and Fatehpur Sikri (Uttar Pradesh)."
        }
        
        # Clean and normalize the query
        clean_query = query.lower().strip()
        
        # Check if we have a direct match in our knowledge base
        for key, value in knowledge_base.items():
            if clean_query in key or key in clean_query:
                return f"Search result: {value}"
        
        # If no direct match, try to use a web API
        try:
            # Using Wikipedia API as a fallback
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                if "extract" in data:
                    return f"Search result: {data['extract']}"
            
            # If Wikipedia fails, try DuckDuckGo
            url = f"https://api.duckduckgo.com/?q={query}&format=json&skip_disambig=1"
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            
            if response.status_code == 200:
                data = response.json()
                if data.get("Abstract"):
                    return f"Search result: {data['Abstract']}"
                elif data.get("RelatedTopics") and len(data["RelatedTopics"]) > 0:
                    topics = data["RelatedTopics"][:3]
                    results = []
                    for topic in topics:
                        if "Text" in topic:
                            results.append(topic["Text"])
                    
                    if results:
                        return "Search results:\n- " + "\n- ".join(results)
            
            return f"I don't have specific information about '{query}'. Please try a different search term or question."
            
        except Exception as e:
            # Fallback to general knowledge
            return f"I don't have specific information about '{query}'. Please try a different search term or question."
    
    return [
        Tool(
            name="Calculator",
            func=calculator,
            description="Useful for mathematical calculations. Input should be a valid mathematical expression."
        ),
        Tool(
            name="TextLength",
            func=text_length,
            description="Count the number of characters in a given text."
        ),
        Tool(
            name="DateTime",
            func=get_date_time,
            description="Get current date and time. Optionally specify a timezone like 'UTC', 'EST', etc."
        ),
        Tool(
            name="Weather",
            func=get_weather,
            description="Get current weather for a location. Input should be a city name or location."
        ),
        Tool(
            name="WebSearch",
            func=search_web,
            description="Search the web for information. Input should be a search query."
        )
    ]

def create_agent():
    """Create and return the LangChain agent with memory"""
    llm = setup_gemini_llm()
    tools = create_sample_tools()
    
    # Create memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Get the ReAct prompt from LangChain hub
    prompt = hub.pull("hwchase17/react")
    
    # Create the agent
    agent = create_react_agent(llm, tools, prompt)
    
    # Create agent executor with memory
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor

def main():
    """Main function to run the agent"""
    try:
        agent = create_agent()
        
        print("🤖 Gemini Agent is ready!")
        print("Type 'quit' to exit\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            try:
                # Add delay to avoid rate limiting
                time.sleep(1)
                print("Agent thinking...")
                response = agent.invoke({"input": user_input})
                print(f"\nAgent: {response['output']}\n")
            except Exception as e:
                print(f"Error: {str(e)}")
                if "429" in str(e) or "quota" in str(e).lower():
                    print("Hit API rate limit. Waiting 10 seconds before trying again...")
                    time.sleep(10)
                print()
                
    except Exception as e:
        print(f"Failed to initialize agent: {str(e)}")
        print("Make sure you have set your GOOGLE_API_KEY in the .env file")

if __name__ == "__main__":
    main()