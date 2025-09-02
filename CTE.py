import numpy as np
import os
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y, check_array
from scipy.optimize import minimize
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
import xgboost as xgb
import json
import torch
from transformers import AutoTokenizer, AutoModel
from FlagEmbedding import BGEM3FlagModel
import argparse

from SAR import TextClassifier

class PCAWrapper:
    def __init__(self, n_components=32):
        self.n_components = n_components
        self.pca = None  # To store the trained PCA model

    def train(self, X):
        self.pca = PCA(n_components=self.n_components)
        X_reduced = self.pca.fit_transform(X)
        return X_reduced

    def transform(self, X):
        if self.pca is None:
            raise ValueError("PCA model has not been trained. Call `train` first.")
        X_reduced = self.pca.transform(X)
        return X_reduced

class TextEncoder:
    def __init__(self, model_name="BAAI/bge-m3", embedding_type="encode", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device(device)
        self.embedding_type = embedding_type
        if embedding_type == 'encode':
            self.model = BGEM3FlagModel(model_name, device=self.device)  # for bge-m3 encode, have no .to(device)
        else:
            self.model = AutoModel.from_pretrained(model_name).to(self.device)  # for bge-m3 last_hidden_state

    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        if self.embedding_type == 'encode':
            with torch.no_grad():
                embedding = self.model.encode(text)['dense_vecs']  # (1024,)
            return embedding.cpu().numpy()
        elif self.embedding_type == 'last_hidden_state':
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state.squeeze().cpu().numpy()
        elif self.embedding_type == 'cls':
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        else:
            raise ValueError("Unsupported embedding type")

def load_dataset(file_path, encoder, embedding_type="encode"):
    if not file_path.endswith(".json"):
        file_path += ".json"  # Add .json extension

    with open(file_path, "r") as f:
        data = json.load(f)

    embeddings = []
    labels = []
    crits = []
    conds = []

    for item in data:
        text = item["text"]
        label = item["label"]
        embedding = item["embedding"]
        crit = item["crit"]
        cond = item["cond"]

        # If embedding is empty, generate it
        if not embedding:
            embedding = encoder.encode(text, embedding_type=embedding_type)

        embeddings.append(np.array(embedding))
        labels.append(label)
        crits.append(crit)
        conds.append(cond)

    return np.array(embeddings), np.array(labels), np.array(crits), np.array(conds)

def find_constant_threshold(y_train, crit_train):
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_train, -crit_train)  # negative for confidence on human

    # Calculate Youden's J statistic
    youden_j = tpr - fpr
    best_index = youden_j.argmax()
    best_threshold = -thresholds[best_index]  # reverse back to positive

    print(f"Constant optimal threshold found: {best_threshold} with Youden's J statistic: {youden_j[best_index]}")
    return best_threshold

def count_probability(input_crit, crit_train, labels_train, k=10):
    # Calculate the distance between the input crit and each crit in the training set
    distances = np.abs(crit_train - input_crit)  # Use absolute distance

    nearest_indices = np.argsort(distances)[:k]  # Find the nearest k crits

    # Count the labels of the nearest k crits
    nearest_labels = labels_train[nearest_indices]
    ai_probability = 1 - np.mean(nearest_labels)  # Calculate the proportion of AI labels

    return ai_probability

def sigmoid(z):
    return expit(z)

def logistic_loss(beta, X, y, crit, sample_weight=None):
    # Calculate the linear part, with a fixed coefficient of -1 for crit
    linear_part = np.dot(X, beta) - crit
    predictions = sigmoid(linear_part)
    
    # Cross-entropy loss
    epsilon = 1e-15  # Prevent log(0)
    loss = -y * np.log(predictions + epsilon) - (1 - y) * np.log(1 - predictions + epsilon)

    # If sample weights are provided, weight the loss
    if sample_weight is not None:
        loss = loss * sample_weight
    
    return np.mean(loss)

class CustomLogisticRegression:
    def __init__(self):
        self.beta = None  # Model parameters (including intercept)
        

    def fit(self, X, y, crit, X_test, y_test, crit_test, class_weight=None):
        # Add a column of all ones to X for the intercept term
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        n_features = X.shape[1]
        beta_init = np.zeros(n_features)  # Initialize beta parameters including intercept

        # Calculate sample weights
        if class_weight == "balanced":
            # Calculate weight for each class: total samples / (number of classes * samples per class)
            unique_classes, class_counts = np.unique(y, return_counts=True)
            class_weights = {cls: len(y) / (len(unique_classes) * count) for cls, count in zip(unique_classes, class_counts)}
            sample_weight = np.array([class_weights[label] for label in y])
        elif isinstance(class_weight, dict):
            # If dictionary format class_weight is provided, use it directly
            sample_weight = np.array([class_weight[label] for label in y])
        else:
            # If class_weight is not specified, don't use sample weights
            sample_weight = None

        # Validation-based parameter selection
        best_beta = None
        best_val_loss_lr = float("inf")
        def callback(beta):
            nonlocal best_beta, best_val_loss_lr  # Declare as external scope variables
            # If validation set is provided, calculate validation loss
            if X_test is not None and y_test is not None and crit_test is not None:
                val_loss = logistic_loss(beta, np.hstack([np.ones((X_test.shape[0], 1)), X_test]), y_test, crit_test)
                if val_loss < best_val_loss_lr:
                    best_val_loss_lr = val_loss
                    best_beta = beta.copy()
        # Optimize logistic regression loss function
        result = minimize(
            logistic_loss,
            beta_init,
            args=(X, y, crit, sample_weight),
            method="CG",
            options={"maxiter": 1000},
            callback=callback
        )
        # Save parameters with best validation performance
        self.beta = best_beta if best_beta is not None else result.x

    def predict(self, X, crit):
        # Calculate linear part, explicitly adding intercept term
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        linear_part = np.dot(X, self.beta) - crit
        predictions = (sigmoid(linear_part) > 0.5).astype(int)  # > 0.5, human, label=1
        return predictions

    def predict_proba(self, X, crit):
        # Calculate linear part, explicitly adding intercept term
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        linear_part = np.dot(X, self.beta) - crit
        probabilities = sigmoid(linear_part)
        return 1 - probabilities[0][0]  # Return AI probability

class CustomXGBoost:
    def __init__(self, max_depth=6, learning_rate=0.1, n_estimators=100):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.model = None

    def fit(self, X, y, crit, X_test, y_test, crit_test, class_weight=None):
        # Check input
        X, y = check_X_y(X, y)

        # Add crit as an additional feature to X with coefficient fixed at -1
        X_with_crit = np.hstack([X, -crit.reshape(-1, 1)])  # crit coefficient fixed at -1

        # Calculate sample weights
        if class_weight == "balanced":
            # Calculate weight for each class: total samples / (number of classes * samples per class)
            unique_classes, class_counts = np.unique(y, return_counts=True)
            class_weights = {cls: len(y) / (len(unique_classes) * count) for cls, count in zip(unique_classes, class_counts)}
            sample_weight = np.array([class_weights[label] for label in y])
        elif isinstance(class_weight, dict):
            # If dictionary format class_weight is provided, use it directly
            sample_weight = np.array([class_weight[label] for label in y])
        else:
            # If class_weight is not specified, don't use sample weights
            sample_weight = None

        # Initialize XGBoost model
        self.model = xgb.XGBClassifier(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
        )

        # Train with validation-based parameter selection
        if X_test is not None and y_test is not None:
            X_val_with_crit = np.hstack([X_test, -crit_test.reshape(-1, 1)])
            eval_set = [(X_with_crit, y), (X_val_with_crit, y_test)]
        else:
            eval_set = [(X_with_crit, y)]
        self.model.fit(
            X_with_crit, y,
            sample_weight=sample_weight,
            verbose=False,
        )

    def predict(self, X, crit):
        # Check input
        X = check_array(X)

        # Add crit as an additional feature to X with coefficient fixed at -1
        X_with_crit = np.hstack([X, -crit.reshape(-1, 1)])

        # Predict labels
        predictions = self.model.predict(X_with_crit)
        return predictions

    def predict_proba(self, X, crit):
        # Check input
        X = check_array(X)

        # Add crit as an additional feature to X with coefficient fixed at -1
        X_with_crit = np.hstack([X, -crit.reshape(-1, 1)])

        # Predict probabilities
        probabilities = 1 - self.model.predict_proba(X_with_crit)
        return probabilities[0][1]

def train_prob_estimator(X_train, X_test, y_train, y_test, crit_train, crit_test, estimator="LogisticRegression", class_weight="balanced"):
    # Initialize the custom model
    if estimator == "LogisticRegression":
        model = CustomLogisticRegression()
        model.fit(X_train, y_train, crit_train, X_test, y_test, crit_test, class_weight=class_weight)
    elif estimator == "XGBoost":
        model = CustomXGBoost(max_depth=6, learning_rate=0.1, n_estimators=100)
        model.fit(X_train, y_train, crit_train, X_test, y_test, crit_test, class_weight=class_weight)

    return model

def evaluate_one_text(input_text, input_crit, input_embedding, input_cond, classifier, datasets_folder, model_name="BAAI/bge-m3", embedding_type="encode", k_neighbors=100, no_condition=False, no_embedding=False, pca_dim=32, no_router=False):
    # 1. Load pretrained model
    device = "cuda"
    encoder = TextEncoder(model_name, embedding_type, device=device)

    # 2. Classify input text
    category = classifier.predict(input_text)[0][0]

    # 3. Load dataset based on router flag
    if no_router:
        print("Ablation: Router disabled. Loading full data partition.")
        category_file = os.path.join(datasets_folder, category)
        embeddings, labels, crits, conds = load_dataset(category_file, encoder, embedding_type)
    else:
        # Default: Use router to get nearest neighbors
        train_data, val_data = classifier.prepare_data(datasets_folder)
        nearest_data = classifier.get_nearest_subcentroids(input_text, input_embedding, n=16)
        embeddings, labels, crits, conds = nearest_data["embeddings"], nearest_data["labels"], nearest_data["crits"], nearest_data["conds"]

    # 4. Process embedding feature if it is being used
    if not no_embedding:
        if pca_dim == -1:
            print("Ablation: PCA disabled. Using full embeddings.")
            embeddings_reduced = embeddings
        else:
            print(f"Applying PCA with {pca_dim} components.")
            pca_wrapper = PCAWrapper(n_components=pca_dim)
            embeddings_reduced = pca_wrapper.train(embeddings)

    # 5. Construct final feature vector for training
    if no_embedding:
        print("Ablation: Using condition features ONLY.")
        Xs = conds
    elif no_condition:
        print("Ablation: Using embedding features ONLY.")
        Xs = embeddings_reduced
    else: # Default: use both
        Xs = np.concatenate((conds, embeddings_reduced), axis=1)

    # 6. Split dataset
    X_train, X_test, y_train, y_test, crit_train, crit_test = train_test_split(Xs, labels, crits, test_size=0.2, random_state=42)
    print(f"Human ratio: {np.sum(y_train) / len(y_train)}")

    # 7. Train probability estimators
    model_lr = train_prob_estimator(X_train, X_test, y_train, y_test, crit_train, crit_test, estimator="LogisticRegression", class_weight="balanced")

    model_xg = train_prob_estimator(Xs, X_test, labels, y_test, crits, crit_test, estimator="XGBoost", class_weight="balanced")

    # 8. Prepare single input vector for prediction
    input_crit_reshaped = input_crit.reshape(1,1)

    if not no_embedding:
        if pca_dim == -1:
            input_embedding_reduced = input_embedding.reshape(1, -1)
        else:
            input_embedding_reduced = pca_wrapper.transform(input_embedding.reshape(1, -1))

    if no_embedding:
        input_X = input_cond.reshape(1, -1)
    elif no_condition:
        input_X = input_embedding_reduced
    else: # Default
        input_X_cond = input_cond.reshape(1, -1)
        input_X = np.concatenate((input_X_cond, input_embedding_reduced), axis=1)

    # 9. Make predictions for the input text
    lr_result = model_lr.predict(input_X, input_crit_reshaped)[0][0]
    lr_proba = model_lr.predict_proba(input_X, input_crit_reshaped)

    xg_result = model_xg.predict(input_X, input_crit_reshaped)[0]
    xg_proba = model_xg.predict_proba(input_X, input_crit_reshaped)
        
    # 10. Calculate baseline method predictions with SAR
    constant_threshold = find_constant_threshold(labels, crits)
    constant_result = 1 if input_crit < constant_threshold else 0
    constant_proba = 0  # Constant method does not provide probability

    count_proba = count_probability(input_crit, crits, labels, k=k_neighbors)
    count_result = 1 if count_proba < 0.5 else 0

    # 11. Calculate baseline predictions on the full (downsampled) dataset
    full_labels, full_crits = [], []
    for filename in os.listdir(datasets_folder):
        if filename.endswith('.json'):
            category_file = os.path.join(datasets_folder, filename)
            _, part_labels, part_crits, _ = load_dataset(category_file, encoder, embedding_type)
            full_labels.append(part_labels)
            full_crits.append(part_crits)
    full_labels = np.concatenate(full_labels, axis=0)[::4]
    full_crits = np.concatenate(full_crits, axis=0)[::4]

    full_constant_threshold = find_constant_threshold(full_labels, full_crits)
    full_constant_result = 1 if input_crit < full_constant_threshold else 0
    full_constant_proba = 0  # Constant method does not provide probability

    full_count_proba = count_probability(input_crit, full_crits, full_labels, k=k_neighbors)
    full_count_result = 1 if full_count_proba < 0.5 else 0
  
    return (lr_result, lr_proba, xg_result, xg_proba, 
            constant_result, constant_proba, count_result, count_proba, 
            full_constant_result, full_constant_proba, full_count_result, full_count_proba)

def evaluate_file(file_path, result_file_path, datasets_folder, embedding_type="encode", model_name="BAAI/bge-m3", sar_path=None, class_path=None, k_neighbors=100, no_condition=False, no_embedding=False, pca_dim=32, no_router=False):
    # Load dataset
    with open(file_path, "r") as f:
        data = json.load(f)
    
    classifier = TextClassifier(
        model_name=model_name,
        embedding_type=embedding_type,
        device="cuda"
    )
    classifier.load_model(model_name=sar_path, class_names_path=class_path)

    # Initialize statistics results
    results = {
        "logistic": {"y_true": [], "y_pred": [], "ai_prob": []},
        "xg": {"y_true": [], "y_pred": [], "ai_prob": []},
        "constant": {"y_true": [], "y_pred": [], "ai_prob": []},
        "count": {"y_true": [], "y_pred": [], "ai_prob": []},
        "full_constant": {"y_true": [], "y_pred": [], "ai_prob": []},
        "full_count": {"y_true": [], "y_pred": [], "ai_prob": []},
    }

    # Iterate over each term in the dataset
    c=1
    for item in data:
        input_text = item["text"]
        input_crit = np.array(item["crit"])
        input_embedding = np.array(item["embedding"])
        input_cond = np.array(item["cond"])
        true_label = int(np.array(item["label"]))

        # Call evaluate_one_text to get prediction results
        lr_result, lr_proba, xg_result, xg_proba, constant_result, constant_proba, count_result, count_proba, full_constant_result, full_constant_proba, full_count_result, full_count_proba = evaluate_one_text(
            input_text, input_crit, input_embedding, input_cond, classifier, datasets_folder=datasets_folder, k_neighbors=k_neighbors,
            no_condition=no_condition, no_embedding=no_embedding, pca_dim=pca_dim, no_router=no_router
        )
        print(f"Finish {c} / {len(data)} test data!")
        c = c + 1

        # Save prediction results and true labels
        results["logistic"]["y_true"].append(true_label)
        results["logistic"]["y_pred"].append(lr_result)
        results["logistic"]["ai_prob"].append(lr_proba)

        results["xg"]["y_true"].append(true_label)
        results["xg"]["y_pred"].append(xg_result)
        results["xg"]["ai_prob"].append(xg_proba)

        results["constant"]["y_true"].append(true_label)
        results["constant"]["y_pred"].append(constant_result)
        results["constant"]["ai_prob"].append(constant_proba)

        results["count"]["y_true"].append(true_label)
        results["count"]["y_pred"].append(count_result)
        results["count"]["ai_prob"].append(count_proba)

        results["full_constant"]["y_true"].append(true_label)
        results["full_constant"]["y_pred"].append(full_constant_result)
        results["full_constant"]["ai_prob"].append(full_constant_proba)

        results["full_count"]["y_true"].append(true_label)
        results["full_count"]["y_pred"].append(full_count_result)
        results["full_count"]["ai_prob"].append(full_count_proba)

    # Save results to file
    # Conversion function for numpy types
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        elif isinstance(obj, (np.int64, np.int32, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # Handle filenames
    original_filename = os.path.basename(file_path)
    filename_without_ext = os.path.splitext(original_filename)[0]
    result_filename = f"{filename_without_ext}_results.json"
    
    # Ensure directory exists
    os.makedirs(result_file_path, exist_ok=True)
    full_result_path = os.path.join(result_file_path, result_filename)
    
    # Convert and save results
    converted_results = convert_types(results)
    with open(full_result_path, 'w') as f:
        json.dump(converted_results, f, indent=4)
    
    print(f"Results saved to: {full_result_path}")

    # Calculate statistical results for each method
    metrics = {}
    for method, result in results.items():
        y_true = result["y_true"]
        y_pred = result["y_pred"]

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="binary")
        recall = recall_score(y_true, y_pred, average="binary")
        f1 = f1_score(y_true, y_pred, average="binary")

        metrics[method] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

    # Print statistical results
    print("\nFinal Metrics:")
    for method, metric in metrics.items():
        print(f"{method.capitalize()} Method:")
        print(f"  Accuracy: {metric['accuracy']:.4f}")
        print(f"  Precision: {metric['precision']:.4f}")
        print(f"  Recall: {metric['recall']:.4f}")
        print(f"  F1 Score: {metric['f1_score']:.4f}")

    return metrics

def main():
    parser = argparse.ArgumentParser(description="Run CTE evaluation on a test file.")
    # Core arguments
    parser.add_argument("--file_path", type=str, required=True, help="Path to the input test file (e.g., cmv_dataset.json).")
    parser.add_argument("--result_file_path", type=str, required=True, help="Path to the directory to save result files.")
    parser.add_argument("--datasets_folder", type=str, required=True, help="Path to the folder containing training data partitions.")
    parser.add_argument("--sar_path", type=str, required=True, help="Path to the SAR model weights (.pt file).")
    parser.add_argument("--class_path", type=str, required=True, help="Path to the SAR class names JSON file.")
    parser.add_argument("--model_name", type=str, default="BAAI/bge-m3", help="Name of the sentence transformer model to use.")
    parser.add_argument("--embedding_type", type=str, default="encode", help="Type of embedding to use ('encode', 'last_hidden_state', 'cls').")
    parser.add_argument("--k_neighbors", type=int, default=100, help="Number of nearest neighbors for count method.")

    # Ablation study arguments
    parser.add_argument('--no_embedding', action='store_true', help='Ablation: Do not use embedding features.')
    parser.add_argument('--no_condition', action='store_true', help='Ablation: Do not use condition features.')
    parser.add_argument('--pca_dim', type=int, default=32, help='Ablation: Specify PCA dimension. Set to -1 to disable PCA.')
    parser.add_argument('--no_router', action='store_true', help='Ablation: Do not use nearest neighbor router.')

    args = parser.parse_args()

    if args.no_condition and args.no_embedding:
        raise ValueError("Cannot specify both --no_condition and --no_embedding, as it would result in an empty feature vector.")

    evaluate_file(
        file_path=args.file_path,
        result_file_path=args.result_file_path,
        datasets_folder=args.datasets_folder,
        embedding_type=args.embedding_type,
        model_name=args.model_name,
        sar_path=args.sar_path,
        class_path=args.class_path,
        k_neighbors=args.k_neighbors,
        # Pass ablation flags
        no_condition=args.no_condition,
        no_embedding=args.no_embedding,
        pca_dim=args.pca_dim,
        no_router=args.no_router
    )

if __name__ == "__main__":
    main()