import numpy as np
import pandas as pd
from collections import Counter

# 1. Entropy Calculation Logic
def calculate_entropy(column):
    values, counts = np.unique(column, return_counts=True)
    n_samples = float(column.shape[0])
    ent = 0.0
    for count in counts:
        probability = count / n_samples
        if probability > 0:
            ent += -probability * np.log2(probability)
    return ent

# 2. Information Gain Logic
def calculate_information_gain(data, feature_name, threshold, target_name):
    # Split data based on the threshold
    left_split = data[data[feature_name] <= threshold]
    right_split = data[data[feature_name] > threshold]
    
    if left_split.shape[0] == 0 or right_split.shape[0] == 0:
        return 0.0
    
    # Calculate weights
    total_samples = data.shape[0]
    weight_left = len(left_split) / total_samples
    weight_right = len(right_split) / total_samples
    
    # Formula IG equals Parent Entropy minus Weighted Child Entropy
    parent_ent = calculate_entropy(data[target_name])
    child_ent = (weight_left * calculate_entropy(left_split[target_name])) + \
                (weight_right * calculate_entropy(right_split[target_name]))
    
    return parent_ent - child_ent

# 3. Decision Tree 
class DecisionTree:
    def __init__(self, depth=0, max_depth=3, target="target", min_samples=5):
        self.left = None
        self.right = None
        self.feature_key = None
        self.threshold_value = None
        self.max_depth = max_depth
        self.depth = depth
        self.target_column = target
        self.min_samples = min_samples
        self.leaf_value = None

    def find_best_split(self, data):
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        features = [col for col in data.columns if col != self.target_column]
        for feature in features:
            # Optimal threshold search for continuous features
            if not pd.api.types.is_numeric_dtype(data[feature]):
                continue

            unique_vals = np.unique(data[feature].dropna())
            if len(unique_vals) <= 1:
                continue

            # Check midpoints between unique values
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
            for thr in thresholds:
                gain = calculate_information_gain(data, feature, thr, self.target_column)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = thr

        return best_feature, best_threshold, best_gain

    def train(self, data):
        labels = data[self.target_column]
        
        # Check leaf conditions
        if len(np.unique(labels)) == 1 or self.depth >= self.max_depth or len(data) < self.min_samples:
            self.leaf_value = Counter(labels).most_common(1)[0][0]
            return

        # Find the split with highest Information Gain
        feat, val, gain = self.find_best_split(data)

        if feat is None or gain <= 0:
            self.leaf_value = Counter(labels).most_common(1)[0][0]
            return

        # Print Entropy and IG for assignment marks
        indent = "  " * self.depth
        current_ent = calculate_entropy(labels)
        print(f"{indent}Depth {self.depth} Split on {feat} | Entropy: {current_ent:.4f} | IG: {gain:.4f}")

        self.feature_key = feat
        self.threshold_value = val

        left_data = data[data[self.feature_key] <= self.threshold_value]
        right_data = data[data[self.feature_key] > self.threshold_value]

        # Recurse
        self.left = DecisionTree(depth=self.depth + 1, max_depth=self.max_depth, target=self.target_column)
        self.left.train(left_data)

        self.right = DecisionTree(depth=self.depth + 1, max_depth=self.max_depth, target=self.target_column)
        self.right.train(right_data)

    def predict(self, row):
        if self.leaf_value is not None:
            return self.leaf_value
        if row[self.feature_key] <= self.threshold_value:
            return self.left.predict(row)
        return self.right.predict(row)

# 4. Helper function to handle missing values and evaluate
def run_model(df, dataset_name, target_col):
    print(f"\n***** PROCESSING DATASET: {dataset_name} *****")
    df_clean = df.copy()

    # Handle missing values via mean and mode imputation
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    
    # Preprocess target to numeric
    df_clean[target_col] = pd.factorize(df_clean[target_col])[0]
    
    # Train Test Split 80/20
    df_clean = df_clean.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(0.8 * len(df_clean))
    train_df = df_clean[:split_idx]
    test_df = df_clean[split_idx:]

    print(f"Dataset Shape: {df_clean.shape}")
    print(f"Initial Entropy: {calculate_entropy(train_df[target_col]):.6f}")

    # Build Decision Tree
    tree = DecisionTree(max_depth=3, target=target_col)
    tree.train(train_df)

    # Prediction
    predictions = [tree.predict(test_df.iloc[i]) for i in range(len(test_df))]
    actuals = test_df[target_col].values
    accuracy = np.mean(predictions == actuals)

    classes = np.unique(actuals)
    precisions, recalls, f1_scores = [], [], []

    for c in classes:
        TP = np.sum((predictions == c) & (actuals == c))
        FP = np.sum((predictions == c) & (actuals != c))
        FN = np.sum((predictions != c) & (actuals == c))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    # Macro-average
    precision_macro = np.mean(precisions)
    recall_macro = np.mean(recalls)
    f1_macro = np.mean(f1_scores)

    print(f"Model Accuracy for {dataset_name}: {accuracy:.4f}")
    print(f"Model Precision for {dataset_name}: {precision_macro:.4f}")
    print(f"Model Recall for {dataset_name}: {recall_macro:.4f}")
    print(f"Model F1-Score for {dataset_name}: {f1_macro:.4f}")

# 5. Load and Execute on all 3 datasets
try:
    np.random.seed(42)
    
    print("Generating Imbalanced Dataset...")
    n_imb = 3000
    X_imb = np.random.uniform(-1, 1, (n_imb, 12))
    # Only 1 percent are Class 1
    target_imb = np.zeros(n_imb)
    fraud_indices = np.random.choice(np.arange(n_imb), size=int(n_imb * 0.01), replace=False)
    target_imb[fraud_indices] = 1

    imb_cols = {f"var_{i}": X_imb[:, i] for i in range(12)}
    imb_cols["Class"] = target_imb.astype(int)
    pd.DataFrame(imb_cols).to_csv("imbalanced_dataset.csv", index=False)


    real = pd.read_csv("real_dataset.csv", sep=";")
    run_model(real, "Bank Marketing Dataset", "y")

    synthetic = pd.read_csv("synthetic_dataset.csv")
    run_model(synthetic, "Synthetic Dataset", "target")

    imbalanced = pd.read_csv("imbalanced_dataset.csv")
    run_model(imbalanced, "Imbalanced Fraud Dataset", "Class")

except FileNotFoundError as e:
    print(f"Error loading files: {e}")