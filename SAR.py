import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import argparse
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from FlagEmbedding import BGEM3FlagModel

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# SubCentroids implementation
class SubCentroidsHead(nn.Module):
    def __init__(self, num_classes, embedding_dim, num_subcentroids=4, temperature=0.1, gamma=0.999):
        super(SubCentroidsHead, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.num_subcentroids = num_subcentroids
        self.temperature = temperature
        self.gamma = gamma  # momentum update parameter
        
        # Initialize prototypes (num_classes, num_subcentroids, embedding_dim)
        self.prototypes = nn.Parameter(
            torch.zeros(self.num_classes, self.num_subcentroids, self.embedding_dim),
            requires_grad=False
        )
        # Initialize with normalized random values
        nn.init.normal_(self.prototypes, std=0.02)
        self.prototypes.data.copy_(F.normalize(self.prototypes.data, p=2, dim=-1))
        
        # Initialize Q as a parameter (num_classes, num_subcentroids)
        self.Q = nn.Parameter(
            torch.zeros(self.num_classes, self.num_subcentroids),
            requires_grad=False
        )

        # Layer normalization for features and output
        self.feat_norm = nn.LayerNorm(embedding_dim)
        self.mask_norm = nn.LayerNorm(num_classes)
    
    def l2_normalize(self, x):
        return F.normalize(x, p=2, dim=-1)
    
    @staticmethod
    def momentum_update(old_value, new_value, momentum):
        update = momentum * old_value + (1 - momentum) * new_value
        return update
    
    def forward(self, x):
        # Normalize features
        x = self.feat_norm(x)
        x = self.l2_normalize(x)
        
        # Ensure prototypes are normalized
        self.prototypes.data.copy_(self.l2_normalize(self.prototypes.data))
        
        # Compute similarity between features and prototypes
        # x: (batch_size, embedding_dim)
        # prototypes: (num_classes, num_subcentroids, embedding_dim)
        # masks: (batch_size, num_subcentroids, num_classes)
        masks = torch.einsum('nd,kmd->nmk', x, self.prototypes)
        
        # For each class, take the maximum similarity over all subcentroids
        # out_cls: (batch_size, num_classes)
        out_cls = torch.amax(masks, dim=1)
        
        # Apply layer norm
        out_cls = self.mask_norm(out_cls)
        
        return out_cls
    
    def update_subcentroids(self, features, labels):
        # Normalize features
        features = self.feat_norm(features)
        features = self.l2_normalize(features)
        
        # Compute similarity between features and prototypes
        masks = torch.einsum('nd,kmd->nmk', features, self.prototypes)
        
        # Get predictions
        out_cls = torch.amax(masks, dim=1)
        pred_cls = torch.argmax(out_cls, dim=1)
        
        # Create mask for correctly classified samples
        correct_mask = (labels == pred_cls)
        
        # Update centroids for each class
        centroids = self.prototypes.data.clone()
        
        for k in range(self.num_classes):
            # Get samples belonging to class k
            class_mask = (labels == k)
            if not torch.any(class_mask):
                continue
                
            # Get features for class k
            class_features = features[class_mask]
            
            # Get initial assignments (similarity to each subcentroid)
            init_q = masks[class_mask, :, k]  # (num_samples_k, num_subcentroids)
            
            # Apply Sinkhorn to get assignment matrix
            q, indices = self.distributed_sinkhorn(init_q)

            # Get correct prediction mask for class k
            correct_class_mask = correct_mask[class_mask]
            
            # Tile the mask for all subcentroids
            m_k_tile = correct_class_mask.unsqueeze(1).expand(-1, self.num_subcentroids)
            
            # Apply mask to assignments
            m_q = q * m_k_tile.float()
            
            # Apply mask to features
            c_k_tile = correct_class_mask.unsqueeze(1).expand(-1, self.embedding_dim)
            c_q = class_features * c_k_tile.float()
            
            # Compute new subcentroid features
            f = torch.matmul(m_q.transpose(0, 1), c_q)
            
            # Count assignments per subcentroid
            n = torch.sum(m_q, dim=0)
            
            # Update subcentroids with non-zero assignments
            if torch.sum(n) > 0:
                # Normalize the new features
                valid_indices = n > 0
                if torch.any(valid_indices):
                    f_valid = f[valid_indices]
                    f_valid = F.normalize(f_valid, p=2, dim=-1)
                    
                    # Apply momentum update
                    new_value = self.momentum_update(
                        old_value=centroids[k, valid_indices],
                        new_value=f_valid,
                        momentum=self.gamma
                    )
                    centroids[k, valid_indices] = new_value
        
        # Update prototypes
        self.prototypes.data.copy_(F.normalize(centroids, p=2, dim=-1))
    
    @torch.no_grad()
    def distributed_sinkhorn(self, out, sinkhorn_iterations=3, epsilon=0.05):
        Q = torch.exp(out / epsilon).t()  # (num_subcentroids, batch_size)
        B = Q.shape[1]  # batch_size
        K = Q.shape[0]  # num_subcentroids
        
        # Make the matrix sum to 1
        sum_Q = torch.sum(Q)
        if sum_Q > 0:
            Q /= sum_Q
        
        for _ in range(sinkhorn_iterations):
            # Normalize each row: total weight per subcentroid must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K
            
            # Normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B
        
        Q *= B  # The columns must sum to 1 so that Q is an assignment
        Q = Q.t()  # (batch_size, num_subcentroids)
        
        indices = torch.argmax(Q, dim=1)
        Q = F.one_hot(indices, num_classes=Q.shape[1]).float()
        
        return Q, indices

# Dataset class
class TextEmbeddingDataset(Dataset):
    def __init__(self, data, class_names):
        self.embeddings = []
        self.labels = []
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
        
        for item in data:
            embedding = np.array(item['embedding'])
            dataset_name = item['dataset']
            label = self.class_to_idx[dataset_name]
            
            self.embeddings.append(embedding)
            self.labels.append(label)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx], dtype=torch.float), torch.tensor(self.labels[idx], dtype=torch.long)

# Model class
class TextClassifier:
    def __init__(self, model_name, device='cuda', embedding_type='encode'):
        self.model_name = model_name
        self.device = device
        self.embedding_type = embedding_type
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if embedding_type == 'encode':
            self.model = BGEM3FlagModel(model_name, device=device)  # for bge-m3 encode, have no .to(device)
        else:
            self.model = AutoModel.from_pretrained(model_name)  # for bge-m3 last_hidden_state
            self.model.to(device)
        
        # Initialize subcentroids
        self.subcentroids_head = None
        self.class_names = None
    
    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        if self.embedding_type == 'encode':
            with torch.no_grad():
                embedding = self.model.encode(text)['dense_vecs']
        elif self.embedding_type == 'last_hidden_state':
            with torch.no_grad():
                outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        elif self.embedding_type == 'cls':
            with torch.no_grad():
                outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        
        return embedding
    
    def prepare_data(self, datasets_folder):
        all_data = []
        class_names = set()
        self.embeddings = []
        self.labels = []
        self.crits = []
        self.conds = []
        self.classes = []
        
        # Load all datasets
        for filename in os.listdir(datasets_folder):
            if filename.endswith(".json"):
                dataset_name = os.path.splitext(filename)[0]
                file_path = os.path.join(datasets_folder, filename)
                
                # Read JSON file
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract embeddings
                for item in data:
                    embedding = item.get('embedding', None)
                    label = item["label"]
                    crit = item["crit"]
                    cond = item["cond"]
                    clas = dataset_name
                    if embedding is not None:
                        if self.embedding_type == 'cls':
                            # vector = np.array(embedding)[0, :]
                            vector = np.array(embedding)
                        else:
                            vector = np.array(embedding)
                        
                        all_data.append({
                            'embedding': vector,
                            'dataset': dataset_name,
                            'text': item.get('text', '')
                        })
                        class_names.add(dataset_name)
                    
                    self.embeddings.append(np.array(embedding))
                    self.labels.append(label)
                    self.crits.append(crit)
                    self.conds.append(cond)
                    self.classes.append(clas)
        
        self.class_names = sorted(list(class_names))
        logger.info(f"Found {len(self.class_names)} classes: {self.class_names}")
        
        # Split data into train and validation
        train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=42)
        
        return train_data, val_data
    
    def train(self, train_data, val_data, num_epochs=10, batch_size=32, 
              num_subcentroids=4, learning_rate=0.001, output_dir="./model_output"):
        # Create datasets
        train_dataset = TextEmbeddingDataset(train_data, self.class_names)
        val_dataset = TextEmbeddingDataset(val_data, self.class_names)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize subcentroids head
        embedding_dim = train_dataset[0][0].shape[0]
        self.subcentroids_head = SubCentroidsHead(
            num_classes=len(self.class_names),
            embedding_dim=embedding_dim,
            num_subcentroids=num_subcentroids
        ).to(self.device)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(self.subcentroids_head.parameters(), lr=learning_rate)
        
        # Initialize loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_acc = 0.0
        for epoch in range(num_epochs):
            # Training
            self.subcentroids_head.train()
            train_loss = 0.0
            
            for batch_idx, (embeddings, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.subcentroids_head(embeddings)
                
                # Calculate loss
                loss = criterion(logits, labels)  #TODO
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update subcentroids
                with torch.no_grad():
                    self.subcentroids_head.update_subcentroids(embeddings, labels)
                
                train_loss += loss.item()
                
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            val_acc, val_f1, val_loss = self.evaluate(val_loader, criterion)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs(output_dir, exist_ok=True)
                
                # Save prototypes
                torch.save(self.subcentroids_head.state_dict(), 
                           os.path.join(output_dir, f"subcentroids_head_epoch{epoch+1}.pt"))
                
                # Save class names
                with open(os.path.join(output_dir, "class_names.json"), 'w') as f:
                    json.dump(self.class_names, f)
                
            logger.info(f"Saved best model with val_acc: {val_acc:.4f}")
    
    def evaluate(self, dataloader, criterion):
        self.subcentroids_head.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0
        
        with torch.no_grad():
            for embeddings, labels in dataloader:
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.subcentroids_head(embeddings)
                
                # Calculate loss
                loss = criterion(logits, labels)
                total_loss += loss.item()
                
                # Get predictions
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return accuracy, f1, total_loss / len(dataloader)
    
    def load_model(self, model_name, class_names_path):
        # Load class names
        with open(class_names_path, 'r') as f:
            self.class_names = json.load(f)
        
        # Load model state dict
        state_dict = torch.load(model_name)
        
        # Determine embedding dimension from state dict
        embedding_dim = state_dict['prototypes'].shape[-1]
        
        # Initialize and load subcentroids head
        self.subcentroids_head = SubCentroidsHead(
            num_classes=len(self.class_names),
            embedding_dim=embedding_dim,
            num_subcentroids=state_dict['prototypes'].shape[1]
        ).to(self.device)
        
        self.subcentroids_head.load_state_dict(state_dict)
        self.subcentroids_head.eval()
    
    def predict(self, text, top_k=3):
        # Get embedding
        embedding = self.get_embedding(text)
        embedding_tensor = torch.tensor(embedding, dtype=torch.float).unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            logits = self.subcentroids_head(embedding_tensor)
        
        # Get probabilities
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        
        # Get top-k predictions
        top_indices = np.argsort(probs)[::-1][:top_k]
        results = []
        
        total_prob = sum(probs[idx] for idx in top_indices)
        for idx in top_indices:
            dataset_name = self.class_names[idx]
            similarity = probs[idx]
            weight = similarity / total_prob
            results.append((dataset_name, similarity, weight))
        
        print(f'SAR classification results: {results}')
        return results
    
    def predict_batch(self, texts):
        # Get embeddings
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        
        embeddings_tensor = torch.tensor(np.array(embeddings), dtype=torch.float).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            logits = self.subcentroids_head(embeddings_tensor)
        
        # Get predictions
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        
        # Map predictions to class names
        results = [self.class_names[pred] for pred in preds]
        
        return results
    
    def get_nearest_subcentroids(self, text, embedding, n=4):
        # Get embedding
        embedding_tensor = torch.tensor(embedding, dtype=torch.float).unsqueeze(0).to(self.device)
        embeddings_tensor = torch.tensor(np.stack(self.embeddings), dtype=torch.float).to(self.device)

        # Normalize features and Ensure prototypes are normalized
        embedding_tensor = self.subcentroids_head.feat_norm(embedding_tensor)
        embedding_tensor = self.subcentroids_head.l2_normalize(embedding_tensor)
        embeddings_tensor = self.subcentroids_head.feat_norm(embeddings_tensor)
        embeddings_tensor = self.subcentroids_head.l2_normalize(embeddings_tensor)
        self.subcentroids_head.prototypes.data.copy_(self.subcentroids_head.l2_normalize(self.subcentroids_head.prototypes.data))
        
        with torch.no_grad():
            # Calculate similarity with all subcentroids
            masks = torch.einsum('nd,kmd->nmk', embedding_tensor, self.subcentroids_head.prototypes)
            masks = masks.squeeze(0)  # (num_subcentroids, num_classes), n=1

        # Find the nearest n subcentroids
        flat_indices = torch.topk(masks.flatten(), n).indices
        topn_subcentroid_indices = flat_indices // masks.shape[1]  # Subcentroid indices
        topn_class_indices = flat_indices % masks.shape[1]  # Class indices


        # Calculate similarity of all raw data with all subcentroids
        similarities = torch.einsum('nd,kmd->nmk', embeddings_tensor, self.subcentroids_head.prototypes)

        # Find the nearest subcentroid for each data point across all classes
        max_similarities, data_subcentroid_indices = torch.max(similarities, dim=1)  # (num_data, num_classes)
        max_similarity_values, data_class_indices = torch.max(max_similarities, dim=1)  # (num_data)

        # Construct the class index and subcentroid index for the nearest subcentroid of each data point
        data_nearest_subcentroids = [
            (data_idx, data_class_indices[data_idx].item(), data_subcentroid_indices[data_idx, data_class_indices[data_idx]].item())
            for data_idx in range(data_class_indices.shape[0])
        ]

        # Iterate through the target subcentroids and filter the corresponding raw data
        matching_data_ids = []
        for data_idx, class_idx, subcentroid_idx in data_nearest_subcentroids:
            for topn_class_idx, topn_subcentroid_idx in zip(topn_class_indices, topn_subcentroid_indices):
                if class_idx == topn_class_idx.item() and subcentroid_idx == topn_subcentroid_idx.item():
                    matching_data_ids.append(data_idx)

        # Return the corresponding data based on the indices
        nearest_data = {
            "embeddings": np.array([self.embeddings[idx] for idx in matching_data_ids]),
            "labels": np.array([self.labels[idx] for idx in matching_data_ids]),
            "crits": np.array([self.crits[idx] for idx in matching_data_ids]),
            "conds": np.array([self.conds[idx] for idx in matching_data_ids])
        }

        # Print class statistics
        data_classes = np.array([self.classes[idx] for idx in matching_data_ids])
        unique_classes, counts = np.unique(data_classes, return_counts=True)
        class_stats = dict(zip(unique_classes, counts))
        print(f"Class statistics: {class_stats}")

        print(f"Number of nearest data: {nearest_data['labels'].shape[0]}")

        return nearest_data

# Main functions
def train_model(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize classifier
    classifier = TextClassifier(
        model_name=args.model_name,
        device=device,
        embedding_type=args.embedding_type
    )
    
    # Prepare data
    train_data, val_data = classifier.prepare_data(args.datasets_folder)
    logger.info(f"Training with {len(train_data)} samples, validating with {len(val_data)} samples")
    
    # Train model
    classifier.train(
        train_data=train_data,
        val_data=val_data,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_subcentroids=args.num_subcentroids,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir
    )

def test_model(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize classifier
    classifier = TextClassifier(
        model_name=args.model_name,
        device=device,
        embedding_type=args.embedding_type
    )
    
    # Load model
    classifier.load_model(
        model_name=args.subcentroids_model_name,
        class_names_path=args.class_names_path
    )
    
    # Get prediction
    results = classifier.predict(args.input_text, top_k=args.top_k)
    
    # Print results
    for dataset_name, similarity, weight in results:
        print(f"Dataset: {dataset_name}, Similarity: {similarity:.4f}, Weight: {weight:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SubCentroids Text Classification",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="mode", help="train or test")

    # Training arguments
    train_parser = subparsers.add_parser(
        "train",
        help="Train a new model"
    )
    train_parser.add_argument("--model_name", type=str, required=True, help="Path to the BGE-M3 model")
    train_parser.add_argument("--datasets_folder", type=str, required=True, help="Path to the datasets folder")
    train_parser.add_argument("--embedding_type", type=str, default="cls", choices=["encode", "last_hidden_state", "cls"], help="Type of embedding to use")
    train_parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    train_parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    train_parser.add_argument("--num_subcentroids", type=int, default=4, help="Number of subcentroids per class")
    train_parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    train_parser.add_argument("--output_dir", type=str, default="./model_output", help="Output directory for models")
    train_parser.add_argument("--no_cuda", action="store_true", help="Do not use CUDA even if available")

    # Testing arguments
    test_parser = subparsers.add_parser(
        "test",
        help="Test an existing model"
    )
    test_parser.add_argument("--model_name", type=str, required=True, help="Path to the BGE-M3 model")
    test_parser.add_argument("--embedding_type", type=str, default="cls", choices=["encode", "last_hidden_state", "cls"], help="Type of embedding to use")
    test_parser.add_argument("--subcentroids_model_name", type=str, required=True, help="Path to the trained subcentroids model")
    test_parser.add_argument("--class_names_path", type=str, required=True, help="Path to the class names JSON file")
    test_parser.add_argument("--input_text", type=str, required=True, help="Input text to classify")
    test_parser.add_argument("--top_k", type=int, default=3, help="Number of top predictions to show")
    test_parser.add_argument("--no_cuda", action="store_true", help="Do not use CUDA even if available")

    args = parser.parse_args()

    if args.mode == "train":
        train_model(args)
    elif args.mode == "test":
        test_model(args)
    else:
        parser.print_help()
