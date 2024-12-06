FROM python:3.11-slim

RUN pip install gradio loguru openai exa_py python-dotenv
ENV OPENAI_API_KEY=${OPENAI_API_KEY}
ENV OPENAI_API_KEY_GHAR=${OPENAI_API_KEY_GHAR}
ENV EXA_API_KEY=${EXA_API_KEY}
ENV JINA_API_KEY=${JINA_API_KEY}

WORKDIR /app
COPY . .
RUN mkdir -p /app/shared_logs/logs && touch /app/shared_logs/logs/log_text.md
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "ui.py", "--share"]