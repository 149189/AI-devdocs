// 11 - LangChain, Agents & RAG
(function () {
  const content = {
    langchain_overview: `# LangChain Overview

LangChain is an open-source framework for building applications powered by large language models. It provides modular components for chaining LLM calls, managing prompts, connecting to data sources, and building agents.

## Key Concepts

- **Chains**: Sequences of LLM calls and transformations
- **Agents**: LLM-driven decision makers that select tools at runtime
- **Retrievers**: Fetch relevant documents from vector stores
- **Memory**: Maintain conversation state across interactions
- **Tools**: Functions that agents can call (search, calculator, APIs)
- **Callbacks**: Hooks for logging, streaming, and monitoring

## How It Works

\`\`\`python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Simple chain with LCEL (LangChain Expression Language)
llm = ChatOpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}")
])

chain = prompt | llm | StrOutputParser()
result = chain.invoke({"input": "Explain transformers in 3 sentences"})

# Chain with retrieval (RAG)
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
\`\`\`

## Architecture

| Component | Purpose |
|-----------|---------|
| LangChain Core | Base abstractions (prompts, LLMs, parsers) |
| LangChain Community | Third-party integrations |
| LangGraph | Stateful multi-step agent workflows |
| LangServe | Deploy chains as REST APIs |
| LangSmith | Observability, testing, evaluation |

## Evolution

- **2022 (Oct)**: LangChain released by Harrison Chase
- **2023**: Rapid growth; LCEL introduced for declarative chains
- **2024**: LangGraph for complex agent workflows; LangSmith for production
- **2025+**: Focus on production-ready agent infrastructure`,

    langgraph: `# LangGraph

LangGraph is LangChain's framework for building stateful, multi-step AI agent workflows as graphs. It enables complex control flow, human-in-the-loop, and persistent state that simple chains cannot handle.

## Key Concepts

- **State Graph**: Define workflow as nodes (functions) connected by edges (transitions)
- **State**: Shared data structure passed between nodes, persisted across steps
- **Nodes**: Functions that process state (LLM calls, tool execution, logic)
- **Edges**: Conditional or fixed transitions between nodes
- **Checkpointing**: Save and resume workflow state
- **Human-in-the-Loop**: Pause for human approval before continuing

## How It Works

\`\`\`python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated

# Define state schema
class AgentState(TypedDict):
    messages: list
    next_action: str

# Define nodes
def call_llm(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

def use_tool(state: AgentState):
    tool_result = execute_tool(state["messages"][-1])
    return {"messages": state["messages"] + [tool_result]}

def should_continue(state: AgentState):
    last = state["messages"][-1]
    if has_tool_call(last):
        return "tool"
    return "end"

# Build graph
graph = StateGraph(AgentState)
graph.add_node("agent", call_llm)
graph.add_node("tool", use_tool)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tool": "tool", "end": END})
graph.add_edge("tool", "agent")

app = graph.compile()
result = app.invoke({"messages": [user_message]})
\`\`\`

## When to Use LangGraph vs Chains

| Scenario | Use |
|----------|-----|
| Simple prompt -> response | LangChain Chain |
| Linear multi-step pipeline | LangChain Chain |
| Branching logic / loops | LangGraph |
| Agent with tool use | LangGraph |
| Human approval steps | LangGraph |
| Long-running workflows | LangGraph (with checkpointing) |

## Evolution

- **2024 (Jan)**: LangGraph released as separate library
- **2024 (Mid)**: Human-in-the-loop and checkpointing added
- **2024 (Late)**: LangGraph Studio for visual debugging
- **2025+**: Production agent orchestration standard`,

    prompt_chains: `# Prompt Templates & Chains

Prompt Templates and Chains are the building blocks of LLM applications. Templates structure inputs to LLMs, while chains compose multiple processing steps into reusable pipelines.

## Prompt Templates

\`\`\`python
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# Basic template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role} expert."),
    ("user", "{question}")
])

# Few-shot template
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
]
few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=ChatPromptTemplate.from_messages([
        ("human", "{input}"), ("ai", "{output}")
    ]),
    examples=examples,
)
\`\`\`

## LCEL (LangChain Expression Language)

\`\`\`python
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Chain composition with pipe operator
chain = prompt | llm | StrOutputParser()

# Parallel execution
parallel = RunnableParallel(
    summary=summary_chain,
    translation=translation_chain,
    sentiment=sentiment_chain
)

# Conditional routing
from langchain_core.runnables import RunnableBranch
branch = RunnableBranch(
    (lambda x: "code" in x["topic"], code_chain),
    (lambda x: "math" in x["topic"], math_chain),
    default_chain  # fallback
)
\`\`\`

## Chain Patterns

| Pattern | Description | Use Case |
|---------|-------------|----------|
| Sequential | A -> B -> C | Multi-step processing |
| Parallel | A, B, C simultaneously | Multiple analyses |
| Branching | Route based on conditions | Topic-specific handling |
| Map-Reduce | Process chunks, combine | Long document summarization |
| Retry | Auto-retry on failure | Robust API calls |

## Applications

- Multi-step reasoning pipelines
- Document processing workflows
- Structured data extraction
- Multi-model orchestration

## Evolution

- **2022**: LangChain introduces chain concept
- **2023**: LCEL replaces legacy chain classes
- **2024**: Composable runnables become the standard pattern
- **2025+**: Chains integrate with LangGraph for complex workflows`,

    memory_systems: `# Memory Systems

Memory systems allow LLM applications to maintain context across conversations and interactions. They solve the fundamental challenge of LLMs having no built-in persistence between calls.

## Types of Memory

| Type | Description | Use Case |
|------|-------------|----------|
| Buffer | Store full conversation history | Short conversations |
| Window | Keep last N messages | Cost control |
| Summary | LLM summarizes past conversation | Long conversations |
| Entity | Track key entities mentioned | Customer profiles |
| Vector | Embed and retrieve past messages | Semantic recall |
| Knowledge Graph | Store relationships | Complex domain knowledge |

## How It Works

\`\`\`python
# Conversation buffer memory
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True)
memory.save_context(
    {"input": "My name is Alice"},
    {"output": "Nice to meet you, Alice!"}
)
# memory.load_memory_variables({}) returns full history

# Summary memory (compresses history)
from langchain.memory import ConversationSummaryMemory

summary_memory = ConversationSummaryMemory(llm=llm)
# Automatically summarizes old messages to save tokens

# Vector-based memory (semantic search over past messages)
from langchain.memory import VectorStoreRetrieverMemory

vectorstore = Chroma(embedding_function=embeddings)
memory = VectorStoreRetrieverMemory(retriever=vectorstore.as_retriever())
# Retrieves most relevant past interactions
\`\`\`

## Memory Architecture in Production

\`\`\`
Short-term: Current conversation context (buffer/window)
    |
Mid-term: Session summaries (summary memory)
    |
Long-term: User preferences, facts (vector store / database)
    |
Persistent: User profile, settings (traditional database)
\`\`\`

## Applications

- Customer service bots (remembering user context)
- Personal AI assistants (long-term preferences)
- Multi-turn research assistants
- Collaborative coding agents

## Evolution

- **2022**: Basic buffer memory in LangChain
- **2023**: Summary, entity, and vector memory types added
- **2024**: Persistent memory via checkpointing in LangGraph
- **2025+**: Long-term memory as a first-class feature in AI agents`,

    langsmith: `# LangSmith & LLM Observability

LangSmith is LangChain's platform for debugging, testing, evaluating, and monitoring LLM applications. It provides the observability needed to move LLM apps from prototype to production.

## Key Features

- **Tracing**: See every step of chain/agent execution with inputs, outputs, latency
- **Evaluation**: Systematic testing of LLM outputs with custom evaluators
- **Datasets**: Manage test datasets for regression testing
- **Monitoring**: Track production metrics, errors, and costs
- **Annotation**: Human feedback and labeling workflows
- **Playground**: Test prompts and chains interactively

## How It Works

\`\`\`python
# Enable tracing (automatic with API key)
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"
os.environ["LANGCHAIN_PROJECT"] = "my-project"

# All LangChain operations are automatically traced
chain = prompt | llm | parser
result = chain.invoke({"input": "Hello"})
# Trace visible in LangSmith UI with full execution details

# Evaluation
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

def correctness_evaluator(run, example):
    prediction = run.outputs["output"]
    reference = example.outputs["answer"]
    score = llm_judge(prediction, reference)
    return {"score": score}

results = evaluate(
    chain.invoke,
    data="my-test-dataset",
    evaluators=[correctness_evaluator]
)
\`\`\`

## Alternatives

| Tool | Provider | Focus |
|------|----------|-------|
| LangSmith | LangChain | LangChain ecosystem |
| Langfuse | Open source | Framework-agnostic tracing |
| Phoenix (Arize) | Arize AI | LLM observability |
| W&B Prompts | Weights & Biases | Experiment tracking |
| Helicone | Open source | Proxy-based logging |

## Evolution

- **2023**: LangSmith launched in beta
- **2024**: GA release with evaluation, annotation, and monitoring
- **2025+**: Production-grade LLM ops platform`,

    rag: `# RAG (Retrieval-Augmented Generation)

RAG combines retrieval systems with generative LLMs to produce accurate, grounded responses by fetching relevant context from external knowledge bases before generating answers.

## Key Concepts

- **Retriever**: Finds relevant documents from a knowledge base
- **Generator**: LLM that produces answers using retrieved context
- **Chunking**: Splitting documents into smaller retrievable units
- **Embedding**: Converting text to vectors for similarity search
- **Grounding**: Ensuring responses are based on retrieved evidence
- **Context Window**: Maximum tokens the LLM can process at once

## How It Works

\`\`\`python
# Basic RAG Pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# 1. Load and chunk documents
loader = PyPDFLoader("knowledge_base.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 2. Embed and store in vector database
vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())

# 3. Retrieve relevant chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
relevant_docs = retriever.get_relevant_documents("What is X?")

# 4. Generate answer with context
prompt = f"Context: {relevant_docs}\\nQuestion: What is X?\\nAnswer:"
response = ChatOpenAI().invoke(prompt)
\`\`\`

## Advanced RAG Techniques

| Technique | Description |
|-----------|-------------|
| Hybrid Search | Combine dense (vector) + sparse (BM25) retrieval |
| Re-ranking | Score and re-order retrieved documents |
| Query Expansion | Rewrite query for better retrieval |
| Multi-Query | Generate multiple query variations |
| Parent-Child | Retrieve child chunks, return parent for context |
| Self-RAG | LLM decides when to retrieve |

## Applications

- Enterprise knowledge bases and Q&A
- Customer support with product documentation
- Legal research and case law analysis
- Medical literature search
- Code documentation assistants

## Evolution

- **2020**: RAG paper published by Facebook AI (Lewis et al.)
- **2022**: LangChain popularizes RAG pipelines
- **2023**: Production RAG systems deployed at scale
- **2024**: Advanced RAG (hybrid, agentic, graph-based) matures
- **2025+**: RAG becomes standard architecture for enterprise LLM apps`,

    vector_databases: `# Vector Databases

Vector databases store and efficiently search high-dimensional vector embeddings. They are the backbone of semantic search, RAG systems, and recommendation engines.

## Key Concepts

- **Embedding**: Dense vector representation of text, images, or other data
- **Similarity Search**: Find vectors closest to a query vector
- **Distance Metrics**: Cosine similarity, Euclidean distance, dot product
- **Approximate Nearest Neighbor (ANN)**: Fast approximate search algorithms
- **Index Types**: HNSW, IVF, PQ for efficient retrieval
- **Metadata Filtering**: Filter by attributes before or during search

## Popular Vector Databases

| Database | Type | Key Features |
|----------|------|-------------|
| Pinecone | Managed cloud | Serverless, auto-scaling |
| Chroma | Open source | Lightweight, embeddable |
| Weaviate | Open source | Hybrid search, multi-modal |
| Qdrant | Open source | Rust-based, high performance |
| Milvus | Open source | Distributed, billion-scale |
| pgvector | PostgreSQL extension | Use existing Postgres |
| FAISS | Library (Meta) | Research-grade, in-memory |

## How It Works

\`\`\`python
# Chroma (simple, local)
import chromadb

client = chromadb.Client()
collection = client.create_collection("docs")
collection.add(
    documents=["AI is transforming healthcare", "ML models learn from data"],
    ids=["doc1", "doc2"]
)
results = collection.query(query_texts=["artificial intelligence"], n_results=2)

# Pinecone (managed, production)
from pinecone import Pinecone
pc = Pinecone(api_key="...")
index = pc.Index("my-index")
index.upsert(vectors=[("id1", [0.1, 0.2, ...], {"text": "..."})])
results = index.query(vector=[0.1, 0.2, ...], top_k=5)
\`\`\`

## Applications

- RAG retrieval for LLM applications
- Semantic search engines
- Image similarity search
- Recommendation systems
- Anomaly detection

## Evolution

- **2017**: FAISS released by Facebook AI for efficient similarity search
- **2021**: Pinecone and Weaviate launch as managed vector databases
- **2022**: Chroma and Qdrant gain popularity with LLM boom
- **2023**: pgvector brings vector search to PostgreSQL
- **2024+**: Vector databases become essential infrastructure for AI applications`,

    embeddings_search: `# Embeddings & Semantic Search

Embeddings convert text, images, and other data into dense vector representations that capture semantic meaning. Semantic search uses these embeddings to find relevant content based on meaning rather than keyword matching.

## Key Concepts

- **Embedding Model**: Neural network that maps inputs to fixed-size vectors
- **Semantic Similarity**: Closer vectors = more similar meaning
- **Cosine Similarity**: cos(a, b) = (a . b) / (|a| * |b|), ranges from -1 to 1
- **Embedding Dimension**: Size of the vector (e.g., 384, 768, 1536)
- **Bi-Encoder**: Encode query and document independently (fast)
- **Cross-Encoder**: Encode query-document pair together (accurate but slow)

## Popular Embedding Models

| Model | Dim | Provider |
|-------|-----|----------|
| text-embedding-3-small | 1536 | OpenAI |
| text-embedding-3-large | 3072 | OpenAI |
| all-MiniLM-L6-v2 | 384 | Sentence Transformers |
| BGE-large-en | 1024 | BAAI |
| E5-mistral-7b | 4096 | Microsoft |
| Cohere embed-v3 | 1024 | Cohere |

## How It Works

\`\`\`python
# Using Sentence Transformers (open source)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode documents and query
docs = ["Machine learning is a subset of AI",
        "Deep learning uses neural networks",
        "The weather is sunny today"]
query = "What is artificial intelligence?"

doc_embeddings = model.encode(docs)
query_embedding = model.encode(query)

# Compute similarities
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
# Result: [0.72, 0.58, 0.08] -> first doc most relevant
\`\`\`

## Applications

- Semantic search (find by meaning, not keywords)
- RAG retrieval for LLM applications
- Duplicate detection
- Clustering and topic modeling
- Cross-lingual search (multilingual embeddings)

## Evolution

- **2013**: Word2Vec introduces word embeddings
- **2018**: Sentence-BERT creates sentence-level embeddings
- **2022**: OpenAI text-embedding-ada-002 becomes popular
- **2024**: Matryoshka embeddings allow variable dimensions
- **2025+**: Multimodal embeddings unify text, image, and code search`,

    ai_agents: `# AI Agents & Tool Use

AI Agents are LLM-powered systems that can reason, plan, and take actions by using external tools. They go beyond simple chatbots by autonomously deciding which actions to take to accomplish goals.

## Key Concepts

- **Agent Loop**: Observe -> Think -> Act -> Observe (repeat)
- **Tool Calling**: LLM generates structured function calls
- **Planning**: Breaking complex tasks into steps
- **Memory**: Storing and retrieving past actions and observations
- **Reflection**: Agent evaluates its own outputs and adjusts
- **Orchestration**: Managing multi-step workflows and tool selection

## How It Works

\`\`\`python
# LangChain ReAct Agent
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun

llm = ChatOpenAI(model="gpt-4")
tools = [DuckDuckGoSearchRun(), WikipediaQueryRun(), calculator_tool]

agent = create_react_agent(llm, tools, prompt_template)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = executor.invoke({
    "input": "What is the population of France divided by the area of Germany?"
})

# Agent execution flow:
# Thought: I need to find population of France
# Action: search("population of France 2024")
# Observation: ~68 million
# Thought: Now I need area of Germany
# Action: search("area of Germany km2")
# Observation: 357,022 km2
# Thought: Now I can calculate
# Action: calculator("68000000 / 357022")
# Observation: ~190.5
# Final Answer: ~190.5 people per km2
\`\`\`

## Agent Frameworks

| Framework | Provider | Approach |
|-----------|----------|----------|
| LangGraph | LangChain | Graph-based agent orchestration |
| CrewAI | CrewAI | Role-based multi-agent teams |
| AutoGen | Microsoft | Multi-agent conversation |
| Semantic Kernel | Microsoft | Enterprise agent framework |
| Claude Agent SDK | Anthropic | SDK for building Claude agents |

## Applications

- Research assistants (search, analyze, summarize)
- Coding assistants (write, test, debug code)
- Data analysis agents (query DBs, create visualizations)
- Customer support with tool access (CRM, ticketing)
- Personal productivity agents

## Evolution

- **2023 (Mar)**: AutoGPT sparks autonomous agent interest
- **2023 (Mid)**: LangChain agents become production-viable
- **2024**: LangGraph, CrewAI, Claude agent SDK for structured agents
- **2025+**: Agents become primary interface for AI applications`,

    function_calling: `# Function Calling

Function Calling (Tool Use) enables LLMs to generate structured outputs that invoke external functions. Instead of free-text responses, the model produces typed JSON arguments for specific functions.

## Key Concepts

- **Function Schema**: JSON schema defining function name, description, and parameters
- **Tool Choice**: LLM decides whether and which function to call
- **Structured Output**: Model produces valid JSON matching the schema
- **Parallel Calls**: Some models call multiple functions simultaneously
- **Forced Calling**: Require the model to use a specific function

## How It Works

\`\`\`python
# OpenAI function calling
import openai

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }
}]

response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools
)

# Model returns: tool_calls=[{name: "get_weather", arguments: {"location": "Paris"}}]
# You execute the function, then send result back to model

# Anthropic tool use
response = anthropic.messages.create(
    model="claude-sonnet-4-20250514",
    tools=[{"name": "get_weather", "description": "...", "input_schema": {...}}],
    messages=[{"role": "user", "content": "Weather in Paris?"}]
)
\`\`\`

## Provider Comparison

| Provider | Feature | Notes |
|----------|---------|-------|
| OpenAI | tools / function_calling | Parallel calls supported |
| Anthropic | tool_use | Streaming tool calls |
| Google | function_declarations | Gemini models |
| Open Source | Varies | Hermes, Gorilla, NexusRaven |

## Applications

- Structured data extraction from unstructured text
- API integration (weather, search, databases)
- Agent tool selection
- Form filling and data entry
- Code execution and REPL integration

## Evolution

- **2023 (Jun)**: OpenAI introduces function calling in GPT API
- **2023 (Nov)**: Anthropic adds tool use to Claude
- **2024**: Parallel function calling, improved reliability
- **2025+**: Function calling becomes standard LLM capability`,

    multi_agent_frameworks: `# Multi-Agent Frameworks

Multi-Agent Frameworks enable multiple AI agents with specialized roles to collaborate on complex tasks. Each agent has specific expertise, tools, and responsibilities within a coordinated workflow.

## Key Concepts

- **Agent Roles**: Specialized agents (researcher, coder, reviewer, etc.)
- **Task Delegation**: Breaking work into sub-tasks assigned to agents
- **Agent Communication**: Agents pass messages and results to each other
- **Orchestration**: Coordinating agent execution order and dependencies
- **Shared Memory**: Common knowledge base accessible to all agents
- **Human-in-the-Loop**: Human oversight and intervention points

## Popular Frameworks

| Framework | Approach | Key Feature |
|-----------|----------|-------------|
| CrewAI | Role-based crews | Simple role/task/crew abstraction |
| AutoGen | Conversational | Agents chat to solve problems |
| LangGraph | Graph-based | Flexible state machine workflows |
| Semantic Kernel | Enterprise | Microsoft ecosystem integration |
| Swarm (OpenAI) | Lightweight | Handoffs between agents |

## How It Works

\`\`\`python
# CrewAI Example
from crewai import Agent, Task, Crew

researcher = Agent(
    role="Research Analyst",
    goal="Find accurate information about {topic}",
    tools=[search_tool, web_scraper],
    llm=ChatOpenAI(model="gpt-4")
)

writer = Agent(
    role="Content Writer",
    goal="Write clear, engaging content based on research",
    llm=ChatOpenAI(model="gpt-4")
)

research_task = Task(
    description="Research {topic} thoroughly",
    agent=researcher,
    expected_output="Detailed research report"
)

writing_task = Task(
    description="Write article based on research",
    agent=writer,
    context=[research_task],
    expected_output="Published article"
)

crew = Crew(agents=[researcher, writer], tasks=[research_task, writing_task])
result = crew.kickoff(inputs={"topic": "AI agents"})
\`\`\`

## Design Patterns

| Pattern | Description | Example |
|---------|-------------|---------|
| Sequential | Agents work one after another | Research -> Write -> Edit |
| Hierarchical | Manager delegates to workers | PM assigns tasks to devs |
| Collaborative | Agents discuss and iterate | Debate, peer review |
| Competitive | Agents propose alternatives | A/B testing ideas |

## Applications

- Software development teams (architect, coder, tester, reviewer)
- Research and analysis workflows
- Content creation pipelines
- Customer support escalation
- Investment research and due diligence

## Evolution

- **2023**: AutoGPT and BabyAGI demonstrate autonomous agents
- **2023 (Oct)**: AutoGen released by Microsoft Research
- **2024 (Jan)**: CrewAI simplifies multi-agent development
- **2024 (Oct)**: OpenAI Swarm for lightweight agent handoffs
- **2025+**: Multi-agent systems become standard for complex AI workflows`,
  };

  Object.assign(window.AI_DOCS, content);
})();
