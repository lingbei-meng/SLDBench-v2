FROM sldbench-runtime:20261016-three-harnesses


# The candidate law runs as this user, never as the grader's. It exists only to
# be unprivileged: it must not be able to write /logs/verifier.
RUN useradd --create-home --uid 1001 --shell /bin/bash grader

# Mount points Harbor fills from the agent's container: /app carries the
# submitted law, /var/lib/oracle the signed spend ledger.
RUN mkdir -p /app /var/lib/oracle

WORKDIR /tests
COPY . /tests
