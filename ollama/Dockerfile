FROM docker.io/ollama/ollama:latest

RUN bash <<EOF
    ollama serve &
    ollama pull nomic-embed-text:latest
    ollama pull gpt-oss:20b
EOF
