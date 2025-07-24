    
# #%%
# graph = 'a-banker'
# model = 'Qwen3-8B'

# # Load the graph and process features
# results_dir = 'results/logit-lens'
# metadata = load_model_results(model, results_dir)

# # Find the specific example in metadata
# target_row = metadata[
#     (metadata['correct_articles'] == 'a') & 
#     (metadata['professions'] == 'banker')
# ].iloc[0]

# correct_article = target_row['correct_articles']
# profession = target_row['professions']

# print(f"Processing: {correct_article}-{profession}")

# # Load the graph file
# model_graph_name = 'qwen3-14b-relu-lowl0'  # Convert from Qwen3-8B to graph directory format
# graph_dir = Path('attribution_graphs') / model_graph_name
# filename = f"{correct_article}-{profession}.pt"
# graph_file = graph_dir / filename

# graph = Graph.from_pt(str(graph_file))

# # Apply graph pruning
# node_threshold = 0.9
# edge_threshold = 0.99

# if node_threshold is None and edge_threshold is None:
#     selected_features = graph.selected_features
# else:
#     pruning_kwargs = {}
#     if node_threshold is not None:
#         pruning_kwargs['node_threshold'] = node_threshold
#     if edge_threshold is not None:
#         pruning_kwargs['edge_threshold'] = edge_threshold
    
#     pruned_graph = prune_graph(graph, **pruning_kwargs)
#     node_mask = pruned_graph.node_mask[:len(graph.selected_features)]
#     selected_features = graph.selected_features[node_mask]

# # Extract unique features (layer, feature_idx pairs)
# no_pos_features_to_pos = defaultdict(set)
# for layer, pos, feature_idx in graph.active_features[selected_features].numpy():
#     layer, pos, feature_idx = int(layer), int(pos), int(feature_idx)
#     no_pos_features_to_pos[(layer, feature_idx)].add((layer, pos, feature_idx))

# print(f"Found {len(no_pos_features_to_pos)} unique features")

# # Get feature texts
# print("Loading feature texts...")
# feature_texts = get_feature_texts_cached(
#     list(no_pos_features_to_pos.keys()),
#     graph.scan
# )

# # Compute similarities using sentence transformer
# print("Computing similarities...")
# similarities = []
# feature_keys = []
# target_word = profession  # "banker"

# model_st = get_sentence_transformer_model('all-MiniLM-L6-v2')
# target_embedding = model_st.encode([target_word])

# for (layer, feature_idx), texts in tqdm(feature_texts.items(), desc="Computing similarities"):
#     if texts:  # Only process if we have texts
#         text_embeddings = model_st.encode([text[17:] for text in texts])
#         # Compute cosine similarities
#         cosine_sims = np.dot(text_embeddings, target_embedding.T).flatten()
#         # Normalize to [0, 1] range
#         cosine_sims = (cosine_sims + 1) / 2
#         max_similarity = float(np.mean(cosine_sims))
#     else:
#         max_similarity = 0.0
    
#     similarities.append(max_similarity)
#     feature_keys.append(f"L{layer}_F{feature_idx}")

# # Convert to numpy arrays for easier manipulation
# similarities = np.array(similarities)
# feature_keys = np.array(feature_keys)

# # Sort by similarity for better visualization
# sort_indices = np.argsort(similarities)[::-1]  # Descending order
# similarities_sorted = similarities[sort_indices]
# feature_keys_sorted = feature_keys[sort_indices]

# print(f"Similarity range: {similarities.min():.3f} - {similarities.max():.3f}")
# print(f"Mean similarity: {similarities.mean():.3f}")
# #%%
# # Plot the similarities
# import matplotlib.pyplot as plt

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# # Plot 1: All similarities (sorted)
# ax1.plot(similarities_sorted, 'b-', alpha=0.7)
# ax1.set_title(f'Feature Similarities to "{target_word}" (All Features, Sorted)', fontsize=14)
# ax1.set_xlabel('Feature Index (sorted by similarity)')
# ax1.set_ylabel('Cosine Similarity')
# ax1.grid(True, alpha=0.3)
# ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Similarity = 0.5')
# ax1.legend()

# # Plot 2: Top 20 most similar features
# top_n = min(50, len(similarities))
# ax2.barh(range(top_n), similarities_sorted[:top_n])
# ax2.set_yticks(range(top_n))
# ax2.set_yticklabels(feature_keys_sorted[:top_n])
# ax2.set_xlabel('Cosine Similarity')
# ax2.set_title(f'Top {top_n} Most Similar Features to "{target_word}"', fontsize=14)
# ax2.grid(True, alpha=0.3, axis='x')

# # Add similarity values as text on bars
# for i in range(top_n):
#     ax2.text(similarities_sorted[i] + 0.01, i, f'{similarities_sorted[i]:.3f}', 
#              va='center', fontsize=10)

# plt.tight_layout()
# plt.show()

# # Print summary statistics
# print(f"\nSummary Statistics:")
# print(f"Total features analyzed: {len(similarities)}")
# print(f"Features with similarity > 0.5: {np.sum(similarities > 0.5)}")
# print(f"Features with similarity > 0.6: {np.sum(similarities > 0.6)}")
# print(f"Features with similarity > 0.7: {np.sum(similarities > 0.7)}")
# #%%
# print(f"\nTop 10 most similar features:")
# for i in range(min(10, len(similarities_sorted))):
#     print(f"{feature_keys_sorted[i]}: {similarities_sorted[i]:.3f}")
#     layer, feature_idx = feature_keys_sorted[i].split('_')
#     layer = int(layer[1:])
#     feature_idx = int(feature_idx[1:])
#     print(feature_texts[(layer, feature_idx)])
# # %%
# layer, feature_idx = feature_keys_sorted[3].split('_')
# layer = int(layer[1:])
# feature_idx = int(feature_idx[1:])
# print(layer, feature_idx)
# print(feature_texts[(layer, feature_idx)])

# feature_texts[layer, feature_idx]

# # %%

# # %%
