FROM node:22-bookworm-slim@sha256:c3de60bf2f9dd0ac6370e6117950ff62d6e339527e7472301c9c78a017978392 AS node
FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258

COPY --from=node /usr/local/bin/node /usr/local/bin/node
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates curl git sudo ripgrep procps tmux asciinema \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-task.txt /tmp/requirements-task.txt
COPY requirements-task-lock.txt /tmp/requirements-task-lock.txt
RUN pip install --no-cache-dir -r /tmp/requirements-task.txt

# prepare.py downloads the official, integrity-checked npm archives.
COPY .cache/claude-code.tgz /tmp/claude-code.tgz
COPY .cache/claude-platform.tgz /tmp/claude-platform.tgz
RUN mkdir -p /usr/local/lib/node_modules/@anthropic-ai/claude-code \
    /usr/local/lib/node_modules/@anthropic-ai/claude-code-linux-x64 \
    && tar -xzf /tmp/claude-code.tgz --strip-components=1 -C /usr/local/lib/node_modules/@anthropic-ai/claude-code \
    && tar -xzf /tmp/claude-platform.tgz --strip-components=1 -C /usr/local/lib/node_modules/@anthropic-ai/claude-code-linux-x64 \
    && node /usr/local/lib/node_modules/@anthropic-ai/claude-code/install.cjs \
    && ln -s /usr/local/lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe /usr/local/bin/claude \
    && chmod +x /usr/local/bin/claude \
    && claude --version \
    && rm /tmp/claude-code.tgz /tmp/claude-platform.tgz
ENV CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_AUTOUPDATER=1

# The same pinned CLIs used by the Codex and Gemini experiment adapters.
# All archives are integrity-checked by prepare.py before the Docker build.
COPY .cache/codex.tgz .cache/codex-platform.tgz .cache/gemini-cli.tgz .cache/node-pty.tgz .cache/node-pty-platform.tgz /tmp/
RUN mkdir -p /opt/sld-cli/node_modules/@openai/codex \
    /opt/sld-cli/node_modules/@openai/codex-linux-x64 \
    /opt/sld-cli/node_modules/@google/gemini-cli \
    /opt/sld-cli/node_modules/@lydell/node-pty \
    /opt/sld-cli/node_modules/@lydell/node-pty-linux-x64 \
    && tar -xzf /tmp/codex.tgz --strip-components=1 -C /opt/sld-cli/node_modules/@openai/codex \
    && tar -xzf /tmp/codex-platform.tgz --strip-components=1 -C /opt/sld-cli/node_modules/@openai/codex-linux-x64 \
    && tar -xzf /tmp/gemini-cli.tgz --strip-components=1 -C /opt/sld-cli/node_modules/@google/gemini-cli \
    && tar -xzf /tmp/node-pty.tgz --strip-components=1 -C /opt/sld-cli/node_modules/@lydell/node-pty \
    && tar -xzf /tmp/node-pty-platform.tgz --strip-components=1 -C /opt/sld-cli/node_modules/@lydell/node-pty-linux-x64 \
    && printf '#!/bin/sh\nexec /usr/local/bin/node /opt/sld-cli/node_modules/@openai/codex/bin/codex.js "$@"\n' > /usr/local/bin/codex \
    && printf '#!/bin/sh\nexec /usr/local/bin/node /opt/sld-cli/node_modules/@google/gemini-cli/bundle/gemini.js "$@"\n' > /usr/local/bin/gemini \
    && chmod +x /usr/local/bin/codex /usr/local/bin/gemini \
    && codex --version && gemini --version \
    && rm /tmp/codex.tgz /tmp/codex-platform.tgz /tmp/gemini-cli.tgz /tmp/node-pty.tgz /tmp/node-pty-platform.tgz
WORKDIR /app
