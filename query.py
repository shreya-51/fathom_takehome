import os
import time
import anthropic

from prompt import MODEL, MAX_TOKENS, TEMPERATURE, SYSTEM_PROMPT

def query_llm(transcript_filepath: str, client) -> str:
    with open(transcript_filepath, "r") as file:
        script = file.read()
        
    response = client.beta.tools.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"extract:\n{script}"
        }]
    )
    
    return response.content[0].text # Only works with 1 message

# query llm 10 times for each episode
def query_all(transcripts_dir: str, output_dir: str, client) -> None:
    transcripts = sorted(os.listdir(transcripts_dir)) # Assuming directory only contains .txt files
    
    for i, transcript in enumerate(transcripts):
        for attempt in range(10): # Running 10 times for each episode
            print(f"Attempt {attempt} for episode {i+1}: {transcript}")
            success = False
            while not success:
                try:
                    llm_response = query_llm(os.path.join(transcripts_dir, transcript), client)
                    success = True
                except anthropic.RateLimitError as error:
                    time.sleep(60)
            
            output_filename = os.path.join(output_dir, f"ep{i+1}_attempt_{attempt}.txt")
            with open(output_filename, "w") as file:
                file.write(llm_response)
            print(f"Response saved in {output_filename}")