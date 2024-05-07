from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score
import matplotlib.pyplot as plt

from evals import extract_json

def character_cluster_analysis(characters_dict, clusters):
    cluster_1 = clusters[0]
    cluster_2 = clusters[1]

    for character in list(characters_dict.keys()):
        # Get all themes the character is a part of
        themes = characters_dict[character]
        cluster_1_count = 0
        cluster_2_count = 0
        for theme in themes:
            # Get which cluster it is a part of
            cluster = get_cluster(theme, cluster_1, cluster_2)
            if get_cluster(theme, cluster_1, cluster_2) == 1:
                cluster_1_count += 1
            else:
                cluster_2_count += 1
        if cluster_1_count > cluster_2_count:
            max_cluster = 1
            percentage = (cluster_1_count / (cluster_1_count + cluster_2_count) ) * 100
        elif cluster_1_count < cluster_2_count:
            max_cluster = 2
            percentage = (cluster_2_count / (cluster_1_count + cluster_2_count) ) * 100
        else:
            max_cluster = 0
            percentage = 50
        
        if max_cluster != 0:
            print(f"{character:20} | mostly in cluster {max_cluster} | percentage: {percentage:6.02f}%")
        else:
            print(f"{character:20} | found equally in each cluster")

def get_themes(filepaths: list[str]):
    themes_dict = {} # key: theme, value: list of characters
    characters_dict = {} # key: character, value: list of themes
    
    for filepath in filepaths:
        with open(filepath, "r") as file:
            response = file.read()
        arcs = extract_json(response)
        
        if "story_arcs" in arcs:
            arcs = arcs["story_arcs"]
        if not isinstance(arcs, list):
            arcs = [arcs]
        
        for arc in arcs:
            characters = arc["characters"]
            themes = arc["themes"]
            for character in characters:
                characters_dict = add_to_dict(characters_dict, themes, character.lower())
            for theme in themes:
                themes_dict = add_to_dict(themes_dict, characters, theme.lower())

    all_themes = list(themes_dict.keys())
    
    return all_themes, themes_dict, characters_dict

def cluster_and_visualize(themes, model):
    theme_embeddings = model.encode(themes)
    kmeans = KMeans(n_clusters=2, random_state=0)
    labels = kmeans.fit_predict(theme_embeddings)

    # Compute clustering metrics
    # silhouette_avg = silhouette_score(theme_embeddings, labels)
    # calinski_harabasz = calinski_harabasz_score(theme_embeddings, labels)
    # print(f"Silhouette Score: {silhouette_avg}")
    # print(f"Calinski-Harabasz Index: {calinski_harabasz}")
    
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(theme_embeddings)
    
    plt.figure(figsize=(5, 4))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Cluster Label')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Theme Clustering Visualization')
    
    plt.show()
    
    # Group themes by their cluster labels
    clusters = {}
    for theme, label in zip(themes, labels):
        if label in clusters:
            clusters[label].append(theme)
        else:
            clusters[label] = [theme]

    return clusters

# Evaluate and plot clustering metrics from n=1 to 10
def evaluate_metrics(themes, model):
    theme_embeddings = model.encode(themes)

    silhouette_scores = []
    calinski_harabasz_scores = []
    k_values = list(range(2, 11))  # K-means clustering requires at least 2 clusters to compute metrics

    for k in k_values:
        if len(theme_embeddings) >= k:  # Ensure there are enough samples for the number of clusters
            kmeans = KMeans(n_clusters=k, random_state=0)
            labels = kmeans.fit_predict(theme_embeddings)

            # Calculate metrics
            if len(set(labels)) > 1:  # More than one cluster
                silhouette_avg = silhouette_score(theme_embeddings, labels)
                calinski_harabasz = calinski_harabasz_score(theme_embeddings, labels)

                silhouette_scores.append(silhouette_avg)
                calinski_harabasz_scores.append(calinski_harabasz)
            else:
                silhouette_scores.append(None)
                calinski_harabasz_scores.append(None)

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(k_values, calinski_harabasz_scores, marker='o')
    plt.title('Calinski-Harabasz Index')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Calinski-Harabasz Index')

    plt.subplot(1, 2, 2)
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.title('Silhouette Score')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')

    plt.tight_layout()
    plt.show()

# ensures everything is lowercase and not duplicated
# note: adding some understanding would make it more robust to themes that are the same but have slightly different wordings
def add_to_dict(the_dict: dict, list_to_add: list, key: str):
    list_to_add = [l.lower() for l in list_to_add]
    if key in the_dict:
        for item in list_to_add:
            if item not in the_dict[key]:
                the_dict[key].append(item)
    else:
        the_dict[key] = list_to_add
    return the_dict

def get_cluster(theme, cluster_1, cluster_2) -> int:
    if theme in cluster_1 and not theme in cluster_2:
        return 1
    elif theme in cluster_2 and not theme in cluster_1:
        return 2
    else:
        raise ValueError