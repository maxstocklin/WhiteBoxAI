import pandas as pd
import numpy as np
from xgboost import DMatrix
from collections import defaultdict
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from collections import Counter




def get_feature_path_ranges(model, X_sample, encoders, categorical_columns):
    booster = model.get_booster()
    tree_df = booster.trees_to_dataframe()
    leaf_indices = booster.predict(DMatrix(X_sample), pred_leaf=True)[0]
    sample = X_sample.iloc[0]

    feature_ranges = {}
    for tree_id, leaf_id in enumerate(leaf_indices):
        tree_data_tree = tree_df[tree_df["Tree"] == tree_id]
        if tree_data_tree.empty:
            continue

        all_ids = set(tree_data_tree['ID'])
        referenced_ids = set(tree_data_tree['Yes']) | set(tree_data_tree['No'])
        root_ids = all_ids - referenced_ids
        node = next(iter(root_ids), None)
        if node is None:
            continue

        while True:
            row = tree_data_tree[tree_data_tree["ID"] == node]
            if row.empty:
                break
            row = row.iloc[0]

            if row["Feature"] == "Leaf":
                break

            feat = row["Feature"]
            yes_node = row["Yes"]
            no_node = row["No"]
            val = sample.get(feat, None)

            if val is None:
                break

            if feat in categorical_columns:
                try:
                    le = encoders[feat]
                    threshold = int(float(row["Split"]))
                    all_cats = list(range(len(le.classes_)))
                    left_cats = [i for i in all_cats if i < threshold]
                    right_cats = [i for i in all_cats if i >= threshold]

                    if val < threshold:
                        chosen_cats = le.inverse_transform(left_cats)
                        node = yes_node
                        side = "left"
                    else:
                        chosen_cats = le.inverse_transform(right_cats)
                        node = no_node
                        side = "right"

                    if feat not in feature_ranges:
                        feature_ranges[feat] = {
                            "type": "categorical",
                            "categories": set(chosen_cats),
                            "side": side
                        }
                    else:
                        feature_ranges[feat]["categories"].update(chosen_cats)

                    continue

                except Exception as e:
                    print(f"[⚠️ WARN] Categorical decoding failed for {feat}: {e}")
                    break

            try:
                threshold = float(row["Split"])
            except:
                break

            if feat not in feature_ranges:
                feature_ranges[feat] = {"min": -np.inf, "max": np.inf}

            if val < threshold:
                feature_ranges[feat]["max"] = min(feature_ranges[feat]["max"], threshold)
                node = yes_node
            else:
                feature_ranges[feat]["min"] = max(feature_ranges[feat]["min"], threshold)
                node = no_node

    # Post-process categorical sets
    for feat in feature_ranges:
        if feature_ranges[feat].get("type") == "categorical":
            feature_ranges[feat]["categories"] = sorted(list(feature_ranges[feat]["categories"]))

    return feature_ranges



def get_used_features(feature_ranges, used_features, X_train, y_train, sample, sample_label, shap_dict=None, encoders=None, categorical_columns=None):
    results = []
    X_train_label = X_train[y_train == sample_label]

    for feat in used_features:
        val = sample[feat]
        shap_score = shap_dict.get(feat, None) if shap_dict else None

        # === Handle categorical features ===
        if feat in categorical_columns and feat in feature_ranges and feature_ranges[feat].get("type") == "categorical":
            cat_list = feature_ranges[feat]["categories"]
            enc = encoders[feat]
            cat_encoded = enc.transform(cat_list)

            in_range_mask = X_train_label[feat].isin(cat_encoded)
            coverage_in_class = in_range_mask.mean() * 100

            all_in_range = X_train[feat].isin(cat_encoded)
            positive_rate = (y_train[all_in_range] == sample_label).mean() * 100

            range_str = f"One of: {sorted(cat_list)}"

            results.append({
                "Feature": feat.replace("_", " ").title(),
                "Value": enc.inverse_transform([val])[0],
                "Z-Score (vs class)": "—",
                "Tree Range": range_str,
                "% Class in Range": round(coverage_in_class, 1),
                "% in Range = Class": round(positive_rate, 1),
                "SHAP": round(shap_score, 3) if shap_score else "(n/a)",
                "Used in Trees": "Yes"
            })

        # === Handle numerical features ===
        elif feat in feature_ranges:
            fmin, fmax = feature_ranges[feat]["min"], feature_ranges[feat]["max"]
            in_range_mask = (X_train_label[feat] >= fmin) & (X_train_label[feat] < fmax)
            coverage_in_class = in_range_mask.mean() * 100

            all_in_range = (X_train[feat] >= fmin) & (X_train[feat] < fmax)
            positive_rate = (y_train[all_in_range] == sample_label).mean() * 100

            z_score = (val - X_train_label[feat].mean()) / X_train_label[feat].std()
            range_str = f"[{fmin:.1f}, {fmax:.1f})"

            results.append({
                "Feature": feat.replace("_", " ").title(),
                "Value": round(val, 2),
                "Z-Score (vs class)": round(z_score, 2),
                "Tree Range": range_str,
                "% Class in Range": round(coverage_in_class, 1),
                "% in Range = Class": round(positive_rate, 1),
                "SHAP": round(shap_score, 3) if shap_score else "(n/a)",
                "Used in Trees": "Yes"
            })

        else:
            results.append({
                "Feature": feat.replace("_", " ").title(),
                "Value": val,
                "Z-Score (vs class)": "—",
                "Tree Range": "Not used in path",
                "% Class in Range": None,
                "% in Range = Class": None,
                "SHAP": round(shap_score, 3) if shap_score else "(n/a)",
                "Used in Trees": "No"
            })

    return pd.DataFrame(results)


import numpy as np
from xgboost import DMatrix

def get_individual_tree_logits(booster, dmatrix, n_trees):
    """Compute individual tree margins by differencing cumulative predictions."""
    tree_logits = []
    prev_margin = 0.0
    # Loop from 1 to n_trees (inclusive)
    for i in range(1, n_trees+1):
        # Use iteration_range=(0, i) to get the cumulative margin up to tree i
        current_margin = booster.predict(dmatrix, output_margin=True, iteration_range=(0, i))[0]
        # The contribution from tree i is the difference from the previous cumulative margin
        tree_logits.append(current_margin - prev_margin)
        prev_margin = current_margin
    return np.array(tree_logits)



from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def compute_distance_score(X_train, sample_df, k=5):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    sample_scaled = scaler.transform(sample_df)

    nbrs = NearestNeighbors(n_neighbors=k).fit(X_train_scaled)
    distances, _ = nbrs.kneighbors(sample_scaled)
    avg_distance = distances.mean()

    # Compute benchmark from training data (subset for speed)
    dists_train, _ = nbrs.kneighbors(X_train_scaled[:200])
    thresholds = np.percentile(dists_train.mean(axis=1), [50, 90])

    if avg_distance <= thresholds[0]:
        score = "✅"
    elif avg_distance <= thresholds[1]:
        score = "⚠️"
    else:
        score = "❌"

    return round(avg_distance, 2), score

def compute_rare_path_coverage(model, X_train, sample_df, rare_threshold=20):
    booster = model.get_booster()
    dtrain = DMatrix(X_train)
    dsample = DMatrix(sample_df)

    pred_leaf_train = booster.predict(dtrain, pred_leaf=True)
    pred_leaf_sample = booster.predict(dsample, pred_leaf=True)[0]

    n_trees = pred_leaf_train.shape[1]

    # Count how many training samples go through each leaf (per tree)
    leaf_counts = Counter()
    for tree_id in range(n_trees):
        leaves = pred_leaf_train[:, tree_id]
        for leaf_id in np.unique(leaves):
            count = np.sum(leaves == leaf_id)
            leaf_counts[(tree_id, leaf_id)] = count

    # Check how many rare leaves the sample went through
    rare_hits = 0
    for tree_id, leaf_id in enumerate(pred_leaf_sample):
        support = leaf_counts.get((tree_id, leaf_id), 0)
        if support <= rare_threshold:
            rare_hits += 1

    rare_ratio = rare_hits / n_trees
    score = "✅" if rare_ratio < 0.1 else "⚠️" if rare_ratio < 0.2 else "❌"

    return round(rare_ratio * 100, 2), score


def dump_all_trees_as_json_df(booster):
    import pandas as pd
    import json

    json_trees = booster.get_dump(dump_format='json')
    tree_dfs = []

    for i, tree_json in enumerate(json_trees):
        tree = json.loads(tree_json)
        nodes = []

        def parse_node(node, depth=0, parent_id=None):
            nid = node['nodeid']
            row = {
                'tree': i,
                'node': nid,
                'depth': depth,
                'parent': parent_id,
                'split': node.get('split'),
                'split_condition': node.get('split_condition'),
                'yes': node.get('yes'),
                'no': node.get('no'),
                'missing': node.get('missing'),
                'leaf': node.get('leaf'),
                'gain': node.get('gain'),
                'cover': node.get('cover'),
            }
            nodes.append(row)
            if 'children' in node:
                for child in node['children']:
                    parse_node(child, depth + 1, nid)

        parse_node(tree)
        tree_dfs.append(pd.DataFrame(nodes))

    return pd.concat(tree_dfs, ignore_index=True)

def get_confidence_report(model, X_train, y_train, sample_df, pred_label):
    sample = sample_df.iloc[0]
    confidence_factors = []

    # === 🔮 Prediction probability ===
    proba = model.predict_proba(sample_df)[0][1]
    proba_score = "✅" if proba > 0.75 or proba < 0.25 else "⚠️" if 0.6 < proba < 0.75 or 0.25 < proba < 0.4 else "❌"
    confidence_factors.append(("🔮 Prediction Probability", f"{proba:.2%}", proba_score))

    # === 📦 Category familiarity ===
    cat_status = "✅"
    cat_message = "Passed all tests"
    for col in sample_df.columns:
        if str(sample_df[col].dtype) == "object":
            if sample[col] not in X_train[col].unique():
                cat_status = "❌"
                cat_message = f"Unseen category in '{col}': {sample[col]}"
                break
            freq = (X_train[col] == sample[col]).mean()
            if freq < 0.02 and cat_status != "❌":
                cat_status = "⚠️"
                cat_message = f"Rare category in '{col}': {sample[col]}, freq: {freq:.2%}"
    confidence_factors.append(("📦 Category Familiarity", cat_message, cat_status))

    # 6. 🌍 Distance to Training Samples (anomaly detection)
    avg_dist, dist_score = compute_distance_score(X_train, sample_df)
    confidence_factors.append(("🌍 Distance to Training Samples", f"{avg_dist:.2f}", dist_score))

    # === 🧨 Z-score anomalies ===
    z_alert = "✅"
    z_messages = []
    num_cols = X_train.select_dtypes("number").columns
    for col in num_cols:
        std = X_train[col].std()
        if std == 0: continue
        z_val = (sample[col] - X_train[col].mean()) / std
        if abs(z_val) > 3:
            z_messages.append(f"{col}: {z_val:.2f}")
            z_alert = "⚠️"
    z_message = "; ".join(z_messages) if z_messages else "No anomalies"
    confidence_factors.append(("🧨 Z-Score Anomalies", z_message, z_alert))

    # === 🧠 Top-tree class consensus ===
    booster = model.get_booster()
    pred_leaf_sample = booster.predict(DMatrix(sample_df), pred_leaf=True)[0]
    pred_leaf_train = booster.predict(DMatrix(X_train), pred_leaf=True)
    n_trees = pred_leaf_train.shape[1]

    def get_individual_tree_logits(booster, dmatrix, n_trees):
        score_contribs = booster.predict(dmatrix, pred_contribs=True)[0]
        return score_contribs[:-1]  # remove bias term

    # Top-k trees consensus
    tree_logits = get_individual_tree_logits(booster, DMatrix(sample_df), n_trees)
    top_k = max(1, int(0.1 * n_trees))
    top_indices = np.argsort(-np.abs(tree_logits))[:top_k]
    top_tree_votes = []
    for i in top_indices:
        leaf_id = pred_leaf_sample[i]
        match_mask = pred_leaf_train[:, i] == leaf_id
        labels_in_leaf = y_train[match_mask]
        vote = round(np.mean(labels_in_leaf)) if len(labels_in_leaf) > 0 else -1
        top_tree_votes.append(vote)
    consensus_top = np.mean(np.array(top_tree_votes) == pred_label)
    consensus_score_top = "✅" if consensus_top > 0.5 else "⚠️"
    confidence_factors.append(("🧠 Top-Tree Class Consensus", f"{consensus_top:.2%}", consensus_score_top))

    # === 🌐 Soft Tree Path Similarity ===
    tree_df = booster.trees_to_dataframe()
    node_counts = defaultdict(int)


    for tree_id in range(n_trees):
        train_nodes = pred_leaf_train[:, tree_id]
        unique, counts = np.unique(train_nodes, return_counts=True)
        for nid, cnt in zip(unique, counts):
            node_counts[(tree_id, nid)] = cnt

    path_weight = 0
    total_weight = 0
    for tree_id in range(n_trees):
        leaf_id = pred_leaf_sample[tree_id]
        support = node_counts.get((tree_id, leaf_id), 0)
        path_weight += support
        total_weight += len(X_train)

    avg_support = path_weight / total_weight
    soft_score = "✅" if avg_support > 0.4 else "⚠️" if avg_support > 0.2 else "❌"
    confidence_factors.append(("🌳 Tree Path Similarity", f"{avg_support:.2%}", soft_score))

    # === Soft Tree Path Similarity 22 (Rare Path Detection) ===
    rare_ratio, rare_score = compute_rare_path_coverage(model, X_train, sample_df, rare_threshold=20)
    confidence_factors.append(("🌐 Rare Path Coverage", f"{rare_ratio:.2f}%", rare_score))


# tree emoji    # === 🌳 Tree Path Coverage ===

    # === Final score ===
    counts = {s: sum(1 for _, _, score in confidence_factors if score == s) for s in ["✅", "⚠️", "❌"]}
    if counts["❌"] > 0:
        level = "🔴 Low"
    elif counts["⚠️"] > 1:
        level = "🟡 Medium"
    else:
        level = "🟢 High"

    df_report = pd.DataFrame(confidence_factors, columns=["Factor", "Status", "Score"])
    return df_report, level




# def get_feature_path_ranges(model, X_sample, encoders, categorical_columns):
#     booster = model.get_booster()
#     tree_df = dump_all_trees_as_json_df(booster)  # ✅ replace trees_to_dataframe()

#     leaf_indices = booster.predict(DMatrix(X_sample), pred_leaf=True)[0]
#     sample = X_sample.iloc[0]

#     feature_ranges = {}
#     for tree_id, leaf_id in enumerate(leaf_indices):
#         tree_data_tree = tree_df[tree_df["tree"] == tree_id]
#         if tree_data_tree.empty:
#             continue

#         all_ids = set(tree_data_tree['node'])
#         referenced_ids = set(tree_data_tree['yes']).union(tree_data_tree['no'])
#         root_ids = all_ids - referenced_ids
#         node = next(iter(root_ids), None)
#         if node is None:
#             continue

#         while True:
#             row = tree_data_tree[tree_data_tree["node"] == node]
#             if row.empty:
#                 break
#             row = row.iloc[0]

#             if row["leaf"] is not None:
#                 break

#             feat = row["split"]
#             yes_node = row["yes"]
#             no_node = row["no"]
#             val = sample.get(feat, None)

#             if val is None:
#                 break

#             if feat in categorical_columns:
#                 try:
#                     le = encoders[feat]
#                     threshold = int(float(row["split_condition"]))
#                     all_cats = list(range(len(le.classes_)))
#                     left_cats = [i for i in all_cats if i < threshold]
#                     right_cats = [i for i in all_cats if i >= threshold]

#                     if val < threshold:
#                         chosen_cats = le.inverse_transform(left_cats)
#                         node = yes_node
#                         side = "left"
#                     else:
#                         chosen_cats = le.inverse_transform(right_cats)
#                         node = no_node
#                         side = "right"

#                     if feat not in feature_ranges:
#                         feature_ranges[feat] = {
#                             "type": "categorical",
#                             "categories": set(chosen_cats),
#                             "side": side
#                         }
#                     else:
#                         feature_ranges[feat]["categories"].update(chosen_cats)

#                     continue

#                 except Exception as e:
#                     print(f"[⚠️ WARN] Categorical decoding failed for {feat}: {e}")
#                     break

#             try:
#                 threshold = float(row["split_condition"])
#             except:
#                 break

#             if feat not in feature_ranges:
#                 feature_ranges[feat] = {"min": -np.inf, "max": np.inf}

#             if val < threshold:
#                 feature_ranges[feat]["max"] = min(feature_ranges[feat]["max"], threshold)
#                 node = yes_node
#             else:
#                 feature_ranges[feat]["min"] = max(feature_ranges[feat]["min"], threshold)
#                 node = no_node

#     # Post-process categorical sets
#     for feat in feature_ranges:
#         if feature_ranges[feat].get("type") == "categorical":
#             feature_ranges[feat]["categories"] = sorted(list(feature_ranges[feat]["categories"]))

#     return feature_ranges










"""
import numpy as np
import pandas as pd
from xgboost import DMatrix
from collections import defaultdict

def percentile_score(value, reference_distribution, thresholds=(0.90, 0.98), reverse=False):
    ref = np.asarray(reference_distribution)
    if reverse:
        p = (ref > value).mean()
    else:
        p = (ref < value).mean()
    if p > thresholds[1]:
        return "❌"
    elif p > thresholds[0]:
        return "⚠️"
    else:
        return "✅"

def get_confidence_report(model, X_train, y_train, sample_df, pred_label):
    sample = sample_df.iloc[0]
    confidence_factors = []

    # === 🔮 Prediction probability ===
    proba = model.predict_proba(sample_df)[0][1]
    proba_score = percentile_score(abs(proba - 0.5), np.abs(model.predict_proba(X_train)[:, 1] - 0.5),
                                   thresholds=(0.3, 0.15), reverse=False)
    confidence_factors.append(("🔮 Prediction Probability", f"{proba:.2%}", proba_score))

    # === 📦 Category familiarity ===
    cat_status = "✅"
    cat_message = "Passed all tests"
    for col in sample_df.columns:
        if str(sample_df[col].dtype) == "object":
            if sample[col] not in X_train[col].unique():
                cat_status = "❌"
                cat_message = f"Unseen category in '{col}': {sample[col]}"
                break
            freq = (X_train[col] == sample[col]).mean()
            if freq < 0.02 and cat_status != "❌":
                cat_status = "⚠️"
                cat_message = f"Rare category in '{col}': {sample[col]}, freq: {freq:.2%}"
    confidence_factors.append(("📦 Category Familiarity", cat_message, cat_status))

    # === 🌍 Distance to Training Samples ===
    distances = np.linalg.norm(X_train - sample_df.values, axis=1)
    avg_dist = distances.mean()
    train_dists = np.linalg.norm(X_train[:, np.newaxis] - X_train, axis=2).mean(axis=1)
    dist_score = percentile_score(avg_dist, train_dists, thresholds=(0.90, 0.98), reverse=True)
    confidence_factors.append(("🌍 Distance to Training Samples", f"{avg_dist:.2f}", dist_score))

    # === 🧨 Z-score anomalies ===
    z_alert = "✅"
    z_messages = []
    num_cols = X_train.select_dtypes("number").columns
    for col in num_cols:
        std = X_train[col].std()
        if std == 0:
            continue
        z_val = (sample[col] - X_train[col].mean()) / std
        z_vals_train = ((X_train[col] - X_train[col].mean()) / std).abs()
        z_score = percentile_score(abs(z_val), z_vals_train, thresholds=(0.90, 0.98), reverse=False)
        if z_score != "✅":
            z_messages.append(f"{col}: {z_val:.2f} ({z_score})")
            if z_score == "❌":
                z_alert = "❌"
            elif z_score == "⚠️" and z_alert != "❌":
                z_alert = "⚠️"
    z_message = "; ".join(z_messages) if z_messages else "No anomalies"
    confidence_factors.append(("🧨 Z-Score Anomalies", z_message, z_alert))

    # === 🧠 Top-tree class consensus ===
    booster = model.get_booster()
    pred_leaf_sample = booster.predict(DMatrix(sample_df), pred_leaf=True)[0]
    pred_leaf_train = booster.predict(DMatrix(X_train), pred_leaf=True)
    n_trees = pred_leaf_train.shape[1]

    tree_logits = booster.predict(DMatrix(sample_df), pred_contribs=True)[0][:-1]
    top_k = max(1, int(0.1 * n_trees))
    top_indices = np.argsort(-np.abs(tree_logits))[:top_k]
    top_tree_votes = []
    for i in top_indices:
        leaf_id = pred_leaf_sample[i]
        match_mask = pred_leaf_train[:, i] == leaf_id
        labels_in_leaf = y_train[match_mask]
        vote = round(np.mean(labels_in_leaf)) if len(labels_in_leaf) > 0 else -1
        top_tree_votes.append(vote)
    consensus_top = np.mean(np.array(top_tree_votes) == pred_label)
    consensus_score_top = percentile_score(consensus_top, [1] * len(X_train), thresholds=(0.5, 0.7))
    confidence_factors.append(("🧠 Top-Tree Class Consensus", f"{consensus_top:.2%}", consensus_score_top))

    # === 🌳 Tree Path Similarity ===
    node_counts = defaultdict(int)
    for tree_id in range(n_trees):
        train_nodes = pred_leaf_train[:, tree_id]
        unique, counts = np.unique(train_nodes, return_counts=True)
        for nid, cnt in zip(unique, counts):
            node_counts[(tree_id, nid)] = cnt

    path_weight = sum(node_counts.get((i, pred_leaf_sample[i]), 0) for i in range(n_trees))
    total_weight = len(X_train) * n_trees
    avg_support = path_weight / total_weight

    # Generate avg_support_train for percentile scoring
    avg_support_train = []
    for i in range(len(X_train)):
        support = sum(node_counts.get((tree_id, pred_leaf_train[i][tree_id]), 0)
                      for tree_id in range(n_trees))
        avg_support_train.append(support / len(X_train) / n_trees)

    soft_score = percentile_score(avg_support, avg_support_train, reverse=True)
    confidence_factors.append(("🌳 Tree Path Similarity", f"{avg_support:.2%}", soft_score))

    # === 🌐 Rare Path Coverage ===
    rare_ratio, rare_score = compute_rare_path_coverage(model, X_train, sample_df, rare_threshold=20)
    confidence_factors.append(("🌐 Rare Path Coverage", f"{rare_ratio:.2f}%", rare_score))

    # === 🧮 Final score ===
    counts = {s: sum(1 for _, _, score in confidence_factors if score == s) for s in ["✅", "⚠️", "❌"]}
    if counts["❌"] > 0:
        level = "🔴 Low"
    elif counts["⚠️"] > 1:
        level = "🟡 Medium"
    else:
        level = "🟢 High"

    df_report = pd.DataFrame(confidence_factors, columns=["Factor", "Status", "Score"])
    return df_report, level


"""



















# def percentile_score(value, reference_distribution, thresholds=(0.90, 0.98), reverse=False):
#     """
#     Assign a confidence score based on how extreme `value` is compared to `reference_distribution`.

#     Parameters:
#         value (float): the sample value to evaluate
#         reference_distribution (array-like): the training values to compare against
#         thresholds (tuple): (⚠️ threshold, ❌ threshold) as percentiles (default: 90% and 98%)
#         reverse (bool): if True, higher values are worse (e.g., distance); else, lower values are worse

#     Returns:
#         score (str): one of "✅", "⚠️", or "❌"
#     """
#     ref = np.asarray(reference_distribution)
#     if reverse:
#         p = (ref > value).mean()  # higher = worse
#     else:
#         p = (ref < value).mean()  # lower = worse

#     if p > thresholds[1]:
#         return "❌"
#     elif p > thresholds[0]:
#         return "⚠️"
#     else:
#         return "✅"


# z_val = ...
# z_vals_train = ...  # as array of abs Z scores
# score = percentile_score(abs(z_val), z_vals_train, thresholds=(0.9, 0.98), reverse=False)




# dist_sample = np.linalg.norm(X_train - sample_df.values, axis=1).min()  # or average
# dist_train = ...  # e.g., distance to nearest neighbor for each point in X_train
# score = percentile_score(dist_sample, dist_train, thresholds=(0.9, 0.98), reverse=True)
# train_dists = np.linalg.norm(X_train - sample_df.values, axis=1)
# p = (train_dists < avg_dist).mean()

# if p > 0.98:
#     score = "❌"
# elif p > 0.9:
#     score = "⚠️"
# else:
#     score = "✅"





# score = percentile_score(avg_support, avg_support_train, thresholds=(0.9, 0.98), reverse=False)





# # Compute avg_support for all training points
# train_leaf = booster.predict(DMatrix(X_train), pred_leaf=True)
# node_counts = defaultdict(int)
# for tree_id in range(train_leaf.shape[1]):
#     for nid in np.unique(train_leaf[:, tree_id]):
#         node_counts[(tree_id, nid)] += np.sum(train_leaf[:, tree_id] == nid)

# # Calculate avg_support for all training points
# avg_support_train = []
# for i in range(train_leaf.shape[0]):
#     support = sum(node_counts[(tree_id, train_leaf[i, tree_id])] for tree_id in range(n_trees))
#     avg_support_train.append(support / len(X_train))

# # Then percentile
# percentile = (np.array(avg_support_train) < avg_support).mean()
# if percentile > 0.98:
#     score = "❌"
# elif percentile > 0.9:
#     score = "⚠️"
# else:
#     score = "✅"
