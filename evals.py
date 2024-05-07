import os
import re
import json
import matplotlib.pyplot as plt
from typing import Tuple, Union
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

from utils import read_text_file, get_embeddings, save_embeddings, load_embeddings

MODEL = SentenceTransformer('all-MiniLM-L6-v2')

def level_0_evals(response: str) -> Tuple[bool, Union[dict, None]]:
    arcs = extract_json(response)
    if not arcs:
        return False, {} # if extracting json fails, we fail this eval
    
    # some cleanup based on current LLM behavior 
    # probably worth adding checks into to see if this ever starts to fail (i.e. changing llm behavior, new models, etc.)
    if "story_arcs" in arcs:
        arcs = arcs["story_arcs"]
    if not isinstance(arcs, list):
        arcs = [arcs]
    
    checks = [
        valid_story_arcs(arcs),
        # can add more checks here
    ]
    
    return all(checks), arcs

def level_1_evals(arcs: json, transcript_filepath: str, embeddings_dir: str) -> float:
    # Generate or load input embeddings (if they already exist)
    embeddings_filepath = os.path.join(embeddings_dir, transcript_filepath.split("/")[-1].split(".")[0] + "_embeddings")
    if os.path.exists(embeddings_filepath + ".npy"):
        input_embeddings = load_embeddings(embeddings_filepath)
    else:
        input_embeddings = get_embeddings(read_text_file(transcript_filepath), MODEL)
        save_embeddings(input_embeddings, embeddings_filepath)
    
    # Get output embeddings
    output_texts = []
    for arc in arcs:
        output_texts.append(arc["description"])
    
    output_embeddings = get_embeddings(output_texts, MODEL)
    cosine_dist = cosine_distances(output_embeddings, input_embeddings)
    threshold = 0.435  # Tuning explained in evals_theory.ipynb
    within_threshold_counts = (cosine_dist < threshold).sum(axis=1)
    return sum(within_threshold_counts) / len(within_threshold_counts)

def valid_story_arcs(arcs: str) -> bool:
    required_fields = ["label", "description", "characters", "themes"]
    for arc in arcs:
        for field in required_fields:
            if not field in required_fields or not arc[field]: # Empty list/str evals to False
                return False
    return True

def extract_json(response: str) -> json:
    try:
        json_data = json.loads(response[response.find("{") : response.rfind("}") + 1])
    except:
        # fail if
        #   1. multiple json objects
        #   2. no json objects
        #   3. incorrect json objects
        return None
    return json_data

# evaluate score for some llm response with some transcript
def evaluate(response_filepath: str, embeddings_filepath: str, threshold: float) -> float:
    with open(response_filepath, "r") as file:
        response = file.read()
    
    arcs = extract_json(response)
    
    if "story_arcs" in arcs:
        arcs = arcs["story_arcs"]
    if not isinstance(arcs, list):
        arcs = [arcs]
    
    input_embeddings = load_embeddings(embeddings_filepath)
    output_texts = []
    for arc in arcs:
        output_texts.append(arc["description"])
    
    output_embeddings = get_embeddings(output_texts, MODEL)
    cosine_dist = cosine_distances(output_embeddings, input_embeddings)
    within_threshold_counts = (cosine_dist < threshold).sum(axis=1)
    return sum(within_threshold_counts) / len(within_threshold_counts)

def is_max_at_matching_episode(scores, episode_number):
    max_value = max(scores)
    max_indices = [i for i, score in enumerate(scores) if score == max_value]
    return (episode_number - 1) in max_indices

def plot_evals_per_episode(threshold, filenames):
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 10))  # Adjust size as needed
    fig.suptitle(f'Score Comparison Across Episodes and Transcripts, threshold: {threshold}', fontsize=16)
    axes = axes.flatten()

    for idx, filepath in enumerate(filenames):
        episode_number = int(re.search(r'ep(\d+)', filepath).group(1))
        scores = []
        transcript_numbers = list(range(1, 11))  # Assuming there are 10 transcripts per episode
        
        for transcript_number in transcript_numbers:
            score = evaluate(filepath, f"./embeddings/s01e{transcript_number:02}_embeddings", threshold)
            scores.append(score)
        
        # Determine color based on the max score condition
        plot_color = 'green' if is_max_at_matching_episode(scores, episode_number) else 'red'
        
        ax = axes[idx]
        ax.plot(transcript_numbers, scores, label=f'Episode {episode_number}', color=plot_color)
        ax.scatter(transcript_numbers, scores, color=plot_color)
        ax.set_xticks(transcript_numbers)
        ax.set_xlabel('Transcript Number')
        ax.set_ylabel('Score')
        ax.set_title(f'Episode {episode_number}')
        ax.legend()

    for ax in axes[len(filenames):]:
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()