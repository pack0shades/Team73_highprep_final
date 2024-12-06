import json
import openai
import re
from crewai import Agent, Task
from loguru import logger
from .config import (
    TOOL_SYSTEM_PROMPT,
    MODEL,
    ALLOW_DELEGATION
)
from dotenv import load_dotenv

load_dotenv()

def get_dynamic_agents(document: str) -> list:
    tool_chat_history = [
        {
            "role": "system",
            "content": TOOL_SYSTEM_PROMPT
        }
    ]
    agent_chat_history = []

    user_msg = {
        "role": "user",
        "content": document
    }

    tool_chat_history.append(user_msg)
    agent_chat_history.append(user_msg)

    output = openai.chat.completions.create(
        messages=tool_chat_history,
        model='gpt-4o-mini'
    ).choices[0].message.content


    def parse_tool_call_str(tool_call_str: str):
        pattern = r'</?tool_call>'
        clean_tags = re.sub(pattern, '', tool_call_str)
        
        try:
            tool_call_json = json.loads(clean_tags)
            return tool_call_json
        except json.JSONDecodeError:
            return clean_tags
        except Exception as e:
            print(f"Unexpected error: {e}")
            return "There was some error parsing the Tool's output"


    # Extract tool call strings using regular expressions
    tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
    tool_calls = re.findall(tool_call_pattern, output, re.DOTALL)

    # Initialize a list to store Agent instances
    agents_list = []
    task_list = []

    # Loop over each tool call and extract arguments
    for tool_call in tool_calls:
        tool_call_json = parse_tool_call_str(tool_call)  # Parse the cleaned tool call string
        if isinstance(tool_call_json, dict) and "arguments" in tool_call_json:
            # Extract the 'arguments' dictionary and pass to the Agent class
            agent_args = tool_call_json["arguments"]
            task_args = tool_call_json["task_arguments"]
            agent = Agent(**agent_args, llm=MODEL, allow_delegation=ALLOW_DELEGATION)  # Unpack the dictionary to pass as keyword arguments
            task = Task(**task_args)
            task.async_execution = True
            # task.human_input = True
            task.agent = agent
            agents_list.append(agent)
            task_list.append(task)

    logger.info (f"Agent List: {agents_list}")
    logger.info (f"Task List: {task_list}")
    logger.info (f"agent list length: {len(agents_list)}")
    logger.info (f"task list length: {len(task_list)}")

    return agents_list, task_list   
