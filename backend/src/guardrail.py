from guardrails.hub.guardrails.regex_match.validator.main import RegexMatch
from guardrails.hub.guardrails.toxic_language.validator.main import ToxicLanguage
from guardrails import Guard
from openai import OpenAI
from .config import QUERY_REFINMENT_PROMPT
from dotenv import load_dotenv
import os

load_dotenv()

client= OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# guardrails returns true or false based on the text
def guardrails(response: str) -> bool:
    guard = Guard().use(
        ToxicLanguage(
            validation_method="sentence",
            threshold=0.5
        )    
    )

    return guard.validate(response).validation_passed


# query refinement
def query_refinment(query: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": QUERY_REFINMENT_PROMPT},
            {"role": "user", "content": query},
        ]
    )
    
    return response.choices[0].message.content