FROM ollama/ollama
USER root
RUN apk add --no-cache curl || (apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*)
USER 1000
