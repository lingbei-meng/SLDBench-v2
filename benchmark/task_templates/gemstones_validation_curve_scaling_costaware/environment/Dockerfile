FROM sldbench-runtime:20261016-three-harnesses

# sudo is the only privilege bridge: one pinned rule lets the unprivileged
# agent invoke the root-owned oracle, and nothing else.
RUN apt-get update \
    && apt-get install -y --no-install-recommends sudo \
    && rm -rf /var/lib/apt/lists/*


# The agent runs unprivileged so the answer table and the spend ledger stay
# out of reach. Harbor applies [agent].user to the oracle agent too, so the
# reference solution has to buy through the same meter.
RUN useradd --create-home --uid 1000 --shell /bin/bash agent

WORKDIR /app

# Catalogue: features + FLOPs price, no targets. Agent-readable.
RUN mkdir -p /app/data
COPY data/catalogue.parquet /app/data/catalogue.parquet

# Private answer table, ledger signing key, and the metered reader.
RUN mkdir -p /opt/oracle/private
COPY oracle/answers.parquet /opt/oracle/private/answers.parquet
COPY oracle/ledger.key /opt/oracle/private/ledger.key
COPY oracle/observe.py /opt/oracle/observe.py
COPY oracle/observe /opt/oracle/observe

# Append-only spend ledger, shipped empty so "the file is missing" can only
# mean collection failed -- never "the agent bought nothing". 700 root:root, so
# the agent cannot delete, truncate or forge it; only the root-owned oracle
# appends, and every line it writes is HMAC-signed with ledger.key.
RUN mkdir -p /var/lib/oracle \
    && : > /var/lib/oracle/ledger.jsonl \
    && chown -R root:root /opt/oracle /var/lib/oracle \
    && chmod 700 /opt/oracle/private /var/lib/oracle \
    && chmod 600 /opt/oracle/private/answers.parquet /opt/oracle/private/ledger.key \
    && chmod 600 /var/lib/oracle/ledger.jsonl \
    && chmod 755 /opt/oracle /opt/oracle/observe \
    && chmod 644 /opt/oracle/observe.py

# One rule, one root-owned wrapper, no arguments that could name another
# program. The wrapper -- not the agent's command line -- chooses the
# interpreter flags, so `python3 -I` is not something the caller can drop.
# env_reset and always_set_home are restated here rather than inherited from
# the base image, so a change to its /etc/sudoers cannot quietly reopen
# PYTHONPATH or HOME as a path into the root process.
RUN printf '%s\n' \
      'Defaults:agent env_reset, always_set_home, secure_path="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"' \
      'agent ALL=(root) NOPASSWD: /opt/oracle/observe' \
      > /etc/sudoers.d/oracle \
    && chmod 440 /etc/sudoers.d/oracle \
    && visudo -cf /etc/sudoers.d/oracle

RUN chown -R agent:agent /app
USER agent

