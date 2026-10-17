# Build context for the separate verifier container.
#
# Harbor uses tests/ as the build context when [verifier] environment_mode is
# "separate", so this image -- not the agent's -- owns /tests. It mirrors the
# task's own environment image (same base, same packages) so a candidate law
# runs against the dependencies it was written against.
FROM sldbench-runtime:20261016-three-harnesses


# Mount points Harbor fills from the agent's container.
RUN mkdir -p /app

WORKDIR /tests
COPY . /tests
