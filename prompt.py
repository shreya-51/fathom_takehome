from textwrap import dedent

MODEL = "claude-3-haiku-20240307"
MAX_TOKENS = 1000
TEMPERATURE = 0
_TOOL_DEF = {
            "name": "story_arcs",
            "description": "Correctly extract a list of story arcs from the given episode.",
            "input_schema": {
                "type": "object",
                "properties": {
                "label": {
                    "type": "string",
                    "description": "The label of the story arc."
                },
                "description": {
                    "type": "string",
                    "description": "A brief description of the story arc."
                },
                "characters": {
                    "type": "array",
                    "description": "The main characters involved in the story arc.",
                    "items": {
                    "type": "string"
                    }
                },
                "themes": {
                    "type": "array",
                    "description": "The themes addressed in the story arc.",
                    "items": {
                    "type": "string"
                    }
                }
                },
                "required": ["label", "description", "characters", "themes"]
            }
        }
SYSTEM_PROMPT = dedent(
        f"""
        You must only response in JSON format that adheres to the following schema:

        <JSON_SCHEMA>
        {_TOOL_DEF}
        </JSON_SCHEMA>
        """
    )

# MESSAGES defined in query since it is dependent on script filepath
# [{
#     "role": "user",
#     "content": f"extract:\n{script}"
# }]