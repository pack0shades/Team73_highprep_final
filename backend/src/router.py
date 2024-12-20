from loguru import logger
from .config import (
    ROUTER_PROMPT,
    MODEL
)
from openai import OpenAI
from crewai import Agent, Task
import os
from dotenv import load_dotenv
load_dotenv()


openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_routed_agent(agent_list: list[Agent], task_list: list[Task], query:str)->list:
    '''
    for ith agent:
        agent_role = agent_list[0].role
        goal = task_list[0].goal
        backstory = task_list[0].backstory

        task_description = task_list[0].description
    '''

    # Validate inputs
    if len(agent_list) != len(task_list):
        raise ValueError("The number of agents must match the number of tasks.")

    # Build the context list
    context_items = []
    for i, (agent, task) in enumerate(zip(agent_list, task_list), 1):
        role = agent.role
        goal = agent.goal
        task_description = task.description
        context_items.append(f"{i}.\n Role: {role}\n Goal: {goal} \n Task: {task_description}")
    
    # Combine the context list into a single string
    context_list = "\n\n".join(context_items)

    # Fill in the ROUTER_PROMPT template
    final_prompt = ROUTER_PROMPT.format(
        num_choices=len(agent_list)-1,
        context_list=context_list,
        query_str=query
    )

    # Call the OpenAI API
    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": final_prompt}
        ]
    ).choices[0].message.content

    response = response.split(" ")

    # Extract the selected agent index
    selected_agent = []
    selected_task = []

    for i in response:
        if int(i) < (len(agent_list)-1):    
            selected_agent.append(agent_list[int(i)])
            selected_task.append(task_list[int(i)])

    logger.info(f"Selected agent: {selected_agent}")
    # return selected_agent
    return selected_agent, selected_task
