from dotenv import load_dotenv
from crewai import (
    Agent,
    Task
)
import warnings
from crewai import (
    Agent,
    Task
)
from loguru import logger
from .config import (
    EXTRACTOR_AGENT_PROMPT,
    EXTRACTOR_TASK,
    MODEL,
)

warnings.filterwarnings('ignore')

load_dotenv()
def get_multi_agents(): # makes static agent when dynamic agents are not required
    # extractor agent creation
    extractor_agent = Agent(
        role=EXTRACTOR_AGENT_PROMPT["role"],
        goal=EXTRACTOR_AGENT_PROMPT["goal"],
        backstory=EXTRACTOR_AGENT_PROMPT["backstory"],
        verbose=EXTRACTOR_AGENT_PROMPT["verbose"],
        allow_delegation=EXTRACTOR_AGENT_PROMPT["allow_delegation"],
        llm=MODEL
    )

    # extractor task
    extractor_task = Task(
        description=EXTRACTOR_TASK["description"],
        expected_output=EXTRACTOR_TASK["expected_output"],
        agent=extractor_agent,
        async_execution=True
    )

    agents = []
    tasks = [] 

    agents.extend([extractor_agent])
    tasks.extend([extractor_task])

    logger.info (f"Number of static agents: {len(agents)}")
    logger.info (f"static agents: {agents}")

    return agents, tasks
