"""Context strings passed to kg-gen's `generate(context=...)` parameter.

kg-gen handles its own LLM prompting internally; these short strings only
tell it how to frame the extraction task for each side of the comparison.
"""

SOURCE_CONTEXT = "Source document for factual verification"
RESPONSE_CONTEXT = "LLM generated response to verify"
