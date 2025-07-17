# LangChain + Gemini Agent

A simple AI agent built with LangChain and Google's Gemini model.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Get your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

3. Copy `.env.example` to `.env` and add your API key:
   ```bash
   cp .env.example .env
   ```

4. Run the agent:
   ```bash
   python agent.py
   ```

## Features

- Interactive chat interface
- Built-in calculator tool
- Text analysis tool
- Rate limiting protection
- Streaming responses

## Customizing Your Agent

### 1. Adding New Tools

To add new tools, modify the `create_sample_tools()` function in `agent.py`. Here's an example of adding a weather tool:

```python
def create_sample_tools():
    """Create sample tools for the agent"""
    # Existing tools...
    
    def get_weather(location: str) -> str:
        """Get weather for a location"""
        # You would implement actual weather API call here
        return f"The weather in {location} is sunny and 72°F"
    
    return [
        # Existing tools...
        Tool(
            name="WeatherTool",
            func=get_weather,
            description="Get the current weather for a location. Input should be a city name."
        ),
    ]
```

### 2. Changing the Model

To use a different Gemini model, modify the `setup_gemini_llm()` function:

```python
def setup_gemini_llm():
    """Initialize Gemini LLM with rate limiting"""
    return ChatGoogleGenerativeAI(
        model="gemini-1.0-pro",  # Change model here
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7,  # Adjust creativity (0.0-1.0)
        max_retries=5,
        streaming=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )
```

Available models:
- `gemini-1.5-flash` (fastest, free tier)
- `gemini-1.5-pro` (more capable)
- `gemini-1.0-pro` (older model)

### 3. Adding Memory

To add conversation memory, update the `create_agent()` function:

```python
from langchain.memory import ConversationBufferMemory

def create_agent():
    """Create and return the LangChain agent with memory"""
    llm = setup_gemini_llm()
    tools = create_sample_tools()
    
    # Add memory
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
```

### 4. Customizing the Prompt

To customize how the agent thinks and responds, you can create your own prompt instead of using the hub prompt:

```python
from langchain.prompts import PromptTemplate

# In create_agent():
prompt = PromptTemplate.from_template("""
You are a helpful AI assistant named Gemini. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
""")
```

### 5. Adding Web Search Capability

To add web search, first install the required package:

```bash
pip install langchain-google-search
```

Then add this to your imports:

```python
from langchain_google_search import GoogleSearchAPIWrapper
```

And add this to your tools:

```python
# In create_sample_tools():
search = GoogleSearchAPIWrapper(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    google_cse_id=os.getenv("GOOGLE_CSE_ID")  # Add this to your .env file
)

# Add this to your tools list
Tool(
    name="GoogleSearch",
    func=search.run,
    description="Search Google for recent information. Input should be a search query."
)
```

### 6. Handling Rate Limits

If you're still hitting rate limits, you can adjust the delay between requests:

```python
# In main() function, inside the try/except block:
time.sleep(3)  # Increase this number to add more delay between requests
```

## Advanced Features

For more advanced features like:
- Multi-agent systems
- Document processing
- Structured outputs
- API integrations

Check out the [LangChain documentation](https://python.langchain.com/docs/get_started/introduction).