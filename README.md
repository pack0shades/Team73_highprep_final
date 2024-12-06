<h1>
  PRADA (Pathway RAg with Dynamic Agents)
</h1>
<div align="center">
  <img src="./assets/blacklogo.png" alt="logo" width="400"/>
</div>

PRADA is an rag application which leaverages dynamic agent creation to provide answers to complex queries as well harnessing pathways cutting edge vectorstore to live stream data, PRADA architecturaly creates insitu specialist agents based of the source data. It leaverages OpenAI API, utilizing embeddings and Chat Completion endpoints to craft dynamic, intelligent responses, setting a new benchmark in dynamic RAG with live streaming data.
## Use Cases
- ### Financial Analytics and Market Insights
PRADA empowers analysts and investors by dynamically processing real-time financial data, including stock market trends, corporate earnings, and macroeconomic indicators. By retrieving and synthesizing insights from live knowledge bases, PRADA delivers actionable intelligence tailored to user queries, enabling data-driven decision-making with unparalleled efficiency.

- ### Legal Document Analysis and Compliance Advisory
In the ever-evolving legal landscape, PRADA provides real-time access to relevant regulations, statutes, and case precedents. By automating compliance analysis and offering advisory insights based on the latest legal developments, PRADA reduces the cognitive load for legal professionals, ensuring accurate and efficient handling of complex legal data.

- ### Supply Chain and Logistics Optimization
For industries dependent on real-time data, PRADA optimizes supply chain operations by analyzing inventory levels, delivery schedules, and demand forecasts. Its dynamic adaptability to global supply chain changes minimizes delays, enhances decision-making, and improves operational efficiency.

## Business Impact
- here

## Key Features
- **Insitu Dynamic Agents**: PRADA dynamically creates agents in real-time, tailored to the streamed documents. This ensures efficiency and adaptability, even when document content changes significantly.
- **Real-Time Data streaming**: Seamlessly integrates with Google Drive to stream live data from a specified folder path. Users can also manually upload documents (PDFs or DOCs) through the user interface.
- **Web Search**: When responses require additional context, users can utilize a web search feature to fetch real-time results directly from the web.
- **Fallback strategy**: A secondary API backup ensures system continuity, maintaining reliability even if the primary API fails.
- **Code Modularity**:  PRADA's modular architecture ensures that all components are independent and reusable, facilitated through Docker for seamless deployment and scalability.  
- **User-Friendly UI via Gradio**: PRADA leverages a Gradio-powered interface, making navigation intuitive and document analysis straightforward, so users can focus on their tasks without technical overhead.

## Future Enhancements
- **Integrating Notion**: Extend PRADA's capabilities to work seamlessly with Notion workspaces. This integration will allow users to retrieve, analyze, and interact with documents and databases stored in Notion for enhanced productivity.
- **Multi-Modal Integration**: Extend support to handle diverse data types such as images, videos, and audio, allowing PRADA to cater to a broader range of use cases, including multimedia analysis and processing.
- **Specialised Mathematical Tools**: Equip PRADA with tools for advanced mathematical problem-solving, including symbolic computation, formula generation, and numerical analysis, to cater to fields like engineering and research.

## Demo Video
[![IMAGE ALT TEXT](./assets/Demo_thumbnail.png)](http://www.youtube.com/watch?v=Pas0NeLZp6I "PRADA(Pathway RAg with Dynamic Agents)")

## 
  # PIPELINE HERE
- **Data Sources**: currently google api krr rhe but if kuchh likhne mila to likh denge
## Streaming Pipeline  

- **Data Ingestion and Processing**:  
  Incoming data from Google Drive is seamlessly processed through PRADA's pipeline. The data is then split into smaller, manageable chunks using Pathway's [Unstructured Parser](https://github.com/pathwaycom/pathway/blob/main/python/pathway/xpacks/llm/parsers.py#L77-L230). Updates to the source data are automatically synced with the pipeline, enabling real-time Retrieval-Augmented Generation (RAG).
- **Guardrails**:  
  Text integrity is verified by [Guardrails](https://github.com/guardrails-ai/guardrails/blob/main/guardrails/guard.py), ensuring the reliability and quality of processed information.  

- **Query Refinement**:  
  User queries are passed through Guardrails for refinement, improving their clarity and relevance for agent interpretation. 

- **Contextual Retrieval**  

  - *Context Creation*:  
     - For each chunk, context is retrieved using an LLM and concatenated with the chunk itself.  
     - Sparse embeddings are generated using the [Splade Encoder](https://github.com/pinecone-io/pinecone-text/blob/main/pinecone_text/sparse/splade_encoder.py), while dense embeddings are created and stored in [Pathway's Vector Store](https://github.com/pathwaycom/pathway/blob/main/python/pathway/xpacks/llm/vector_store.py#L628-L746).  

 

  - *Retrieval and Rank Fusion*:  
     - Context is retrieved using Pathway's [KNN Index](https://github.com/pathwaycom/pathway/blob/main/python/pathway/stdlib/ml/index.py#L9-L301).  
     - Rank fusion calculates the harmonic mean of scores derived from both dense and sparse embeddings, ensuring an accurate final score for each document.  

- **Dynamic Agent Generation**:  
  - Dynamic agents are generated using an LLM by leveraging the context of documents stored in the vector store. This process runs parallel to retrieval and rank fusion, ensuring efficiency.  

- **Router**:  
  - Queries are routed only to the most relevant agents, minimizing unnecessary responses and optimizing performance.  

- **Initial Crew Formation**:  
   - A preliminary crew of agents is formed, including:  
     - Selected agents based on the router's recommendation.  
     - A Question-Answering agent.  
     - A meta-agent, which consolidates retrieved documents, user queries, and knowledge from the entire document corpus.  
   - The crew's outputs are processed by the meta-agent to generate an initial response.  

- **Reflection**:  
   - The meta-agent's output is evaluated by a critique agent, providing feedback and suggestions.  
   - The initial crew iteratively incorporates this feedback, refining the response for `n` iterations.  

- **Final Response**:  
   - The meta-agent's final output is verified by Guardrails before being displayed on the user interface, ensuring a reliable and contextually accurate response. 

## Usage

### Creating credentials.json in the Google API console:
- Go to https://console.cloud.google.com/projectcreate and create new project
- Enable Google Drive API by going to https://console.cloud.google.com/apis/library/drive.googleapis.com, make sure the newly created project is selected in the top left corner
- Configure consent screen:
  - Go to https://console.cloud.google.com/apis/credentials/consent
  - If using a private Gmail, select "External", and go next.
  - Fill required parameters: application name, user support, and developer email (your email is fine)
  - On the next screen click "Add or remove scopes" search for "drive.readonly" and select this scope
  - Save and click through other steps
- Create service user:
  - Go to https://console.cloud.google.com/apis/credentials
  - Click "+ Create credentials" and create a service account
  - Name the service user and click through the next steps
- Generate service user key:
  - Once more go to https://console.cloud.google.com/apis/credentials and click on your newly created user (under Service Accounts)
  - Note service user email address, it will be needed later
  - Go to "Keys", click "Add key" -> "Create new key" -> "JSON"
A JSON file will be saved to your computer. Move it to the folder where your Pathway script is located and rename it to credentials.json.


## Installation

### A. Run with Docker

### Prerequisites

Ensure you have Docker and docker compose both latest version installed on your system before proceeding. Docker compose  will be used to build and run the application in a containerized environment. For installation please refer the offcial documneation of docker [Docket Installation Guide](https://docs.docker.com/compose/install/linux/)


- **JinaAI API Key**: 
  - Create an [JinaAI](https://jina.ai/) account and generate a new API Key
  - This key is crucial for searching the query on web through JinaAI.
- **Exa API Key**: 
  - Create an [Exa](https://exa.ai/) account and generate a new API Key
  - This key acts as a backup in case the primary API key for web search fails.
- **Guardrails API Key**: 
  - Create an [GuardrailsAI](https://www.guardrailsai.com/) account and generate a new API Key
  - This key is crucial for searching the query on web through JinaAI.
- **Google Drive Credentials**:
    - A json file containing the credentials as mentioned above is uploaded through the UI.
- **OpenAI API Key**:
    - Create an [OpenAI](https://openai.com/) account and generate a new API Key: 
    - To access the OpenAI API, you will need to create an API Key. You can do this by logging into the [OpenAI] (https://openai.com/product) website and navigating to the API Key management page.


### 1. Environment Setup

Export the following API keys to the source file 

   ```bashrc
   OPENAI_API_KEY={OPENAI_API_KEY}
   JINA_API_KEY={JINA_API_KEY}
   EXA_API_KEY={EXA_API_KEY}
   GUARDRAILS_API_KEY={GUARDRAILS_API_KEY}
   ```

This file will be used by Docker to set the environment variables inside the container.

 