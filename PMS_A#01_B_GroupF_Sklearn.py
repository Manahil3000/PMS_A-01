import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics as sk_metrics
from sklearn.model_selection import train_test_split

# Custom function to pull out split info from the sklearn object
def show_split_details(tree_obj, feature_list):
    """
    This helper traverses the trained tree to print depth and calculate IG manually.
    """
    left_nodes = tree_obj.children_left
    right_nodes = tree_obj.children_right
    node_features = tree_obj.feature
    node_entropy = tree_obj.impurity
    sample_counts = tree_obj.n_node_samples

    def get_node_info(id, current_depth):
        # Indent output based on how deep we are in the tree
        space = "  " * current_depth
        
        # Check if this node actually splits or is a leaf
        if left_nodes[id] != right_nodes[id]:
            fname = feature_list[node_features[id]]
            
            # Weighted entropy of children
            l_child = left_nodes[id]
            r_child = right_nodes[id]
            
            # Probability weights for children
            p_left = sample_counts[l_child] / sample_counts[id]
            p_right = sample_counts[r_child] / sample_counts[id]
            
            # IG = parent entropy - sum(child entropies)
            info_gain = node_entropy[id] - (p_left * node_entropy[l_child] + p_right * node_entropy[r_child])
            
            print(f"{space}Depth {current_depth} -> Splitting on '{fname}' | Entropy: {node_entropy[id]:.4f} | IG: {info_gain:.4f}")
            
            # Go deeper
            get_node_info(l_child, current_depth + 1)
            get_node_info(r_child, current_depth + 1)

    get_node_info(0, 0)

def process_and_test(data, name, target):
    print(f"\n--- Running Sklearn Benchmark for: {name} ---")
    
    if target not in data.columns:
        print(f"Error: Target '{target}' not found. Check your file delimiter (sep=';')!")
        return
    
    # Cleaning data - filling nulls (Requirement: Handle missing values)
    # Using simple loops for a more manual feel
    for col in data.columns:
        if data[col].isnull().any():
            if data[col].dtype in ['int64', 'float64']:
                data[col] = data[col].fillna(data[col].mean())
            else:
                data[col] = data[col].fillna(data[col].mode()[0])
    
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = pd.factorize(data[col])[0]
    
    # Set up X and y
    X_vals = data.drop(target, axis=1)
    y_vals = data[target]
    feat_names = list(X_vals.columns)

    # Standard 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(X_vals, y_vals, test_size=0.2, random_state=42)

    # Initialize the classifier (Requirement: Support Entropy)
    # Max depth set to 3 to keep it comparable to the scratch model
    model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # Print the tree internals for assignment marks
    print(f"Base Entropy at Root: {model.tree_.impurity[0]:.5f}")
    show_split_details(model.tree_, feat_names)

    # Get predictions and score it
    preds = model.predict(X_test)
    acc = sk_metrics.accuracy_score(y_test, preds)

    prec = sk_metrics.precision_score(y_test, preds, average='macro', zero_division=0)
    rec = sk_metrics.recall_score(y_test, preds, average='macro', zero_division=0)
    f1 = sk_metrics.f1_score(y_test, preds, average='macro', zero_division=0)
    
    print(f"Final Accuracy Score: {acc:.4f}")
    print(f"Final Precision Score: {prec:.4f}")
    print(f"Final Recall Score: {rec:.4f}")
    print(f"Final F1-Score: {f1:.4f}")

# Main execution logic
if __name__ == "__main__":
    try:
        # 1. Load the real dataset (Requirement: 1000+ rows)
        df_real = pd.read_csv("real_dataset.csv", sep=";")
        process_and_test(df_real, "Bank Data (Real)", "y")

        # 2. Load synthetic
        df_synth = pd.read_csv("synthetic_dataset.csv")
        process_and_test(df_synth, "Synthetic Data", "target")

        # 3. Load imbalanced
        df_imb = pd.read_csv("imbalanced_dataset.csv")
        process_and_test(df_imb, "Highly Imbalanced Data", "Class")

    except Exception as error:
        print(f"Process failed: {error}")