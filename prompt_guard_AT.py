import logging
from datetime import datetime
import matplotlib.pyplot as plt
import pandas
import seaborn as sns
import time
import torch
import numpy as np
import yaml
import importlib

from datasets import load_dataset
from sklearn.metrics import auc, roc_curve, roc_auc_score
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class AdvPromptGuardTrainer:
    def __init__(self, model, tokenizer, device='cpu', attack=None, gcg_attack=None, config=None, log_file='adv_prompt_guard.log'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.attack = attack
        self.gcg_attack = gcg_attack
        self.config=config

        
        # Input token embeddings should not be trained (especially when they define the threat model)
        model.deberta.embeddings.requires_grad_(False)
        
        self.hard_tokens = self.model.deberta.embeddings.word_embeddings.weight
        
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.attack.logger = self.logger
        
        # Log initialization info
        self.logger.info(f"Initializing AdvPromptGuardTrainer on device: {device}")
        self.logger.info(f"Model: {type(model).__name__}")
        self.logger.info(f"Tokenizer: {type(tokenizer).__name__}")
        self.logger.info("Input token embedding layer frozen")
        
        self.test_distance_calculation()

        
    def distance_to_closest_hard_token(self, emb):
        """Compute Euclidean distance between input embeddings and all hard-token embeddings.
        
        Args:
            emb: Tensor of shape (*, embedding_dim)
            
        Returns:
            min_distances: Tensor of shape (*,) containing squared L2 distances
            closest_token_ids: LongTensor of shape (*,) with closest token IDs
        """
        # Ensure input has at least 2 dimensions for unsqueeze
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
            
        # Calculate squared distances: (..., 1, emb_dim) - (vocab_size, emb_dim)
        square_distances = (emb.unsqueeze(-2) - self.hard_tokens)**2  # (..., vocab_size, emb_dim)
        square_distances = square_distances.sum(dim=-1)  # (..., vocab_size)
        
        # Get closest token IDs and distances
        min_distances, closest_token_ids = torch.min(square_distances, dim=-1)
        return min_distances, closest_token_ids
    
       
    def _align_labels(self, labels):
        """Map dataset labels (0,1) to model labels (0,2)"""
        return torch.tensor([0 if x == 0 else 2 for x in labels], device=self.device)
        
    def evaluate(self, dataset, batch_size=16, attack=None, attack_steps=0, save_embeddings=False):
        """Evaluate model on dataset with optional adversarial attacks
        
        Args:
            dataset: Dataset to evaluate on
            batch_size: Batch size for evaluation
            attack_steps: Number of attack steps (0 for benign evaluation)
            
        Returns:
            scores: Model scores for positive class
            labels: Ground truth labels
            is_adv: Whether each example was adversarially modified
        """
        self.model.eval()
        texts = dataset['text']
        labels = dataset['label']
        
        eval_type = "Benign" if attack_steps == 0 else f"Adversarial (steps={attack_steps})"
        self.logger.info(f"Starting {eval_type} evaluation on {len(texts)} samples")
                
        # Batch evaluation
        scores_soft, scores_hard = [], []
        labels_list, is_adv_list = [], []
        all_embeddings = [] if save_embeddings else None
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            if attack_steps > 0:
                # Generate adversarial examples for positive samples
                adv_emb, attention_mask, input_ids_hard = self.generate_adversarial_batch(
                    batch_texts, batch_labels, attack, attack_steps)
                
                # Directly use the adversarial embeddings
                norm_emb = self.model.deberta.embeddings.LayerNorm(adv_emb)
                outputs = self.model.deberta.encoder(norm_emb, attention_mask)[0]
                logits = self.model.classifier(self.model.pooler(outputs))
                probs = torch.softmax(logits, dim=-1)
                batch_scores_soft = probs[:, 2].detach().cpu().numpy()  # Jailbreak class score
                
                # Hard-token evaluation (using input_ids_hard)
                inputs_hard = {
                    'input_ids': input_ids_hard.to(self.device),
                    'attention_mask': attention_mask.to(self.device)
                }
                with torch.no_grad():
                    logits_hard = self.model(**inputs_hard).logits
                    batch_scores_hard = torch.softmax(logits_hard, dim=-1)[:, 2].cpu().numpy()
                
                scores_soft.extend(batch_scores_soft)
                scores_hard.extend(batch_scores_hard)
                is_adv_list.extend([1 if lbl == 2 else 0 for lbl in batch_labels])
                
                if save_embeddings:
                    batch_emb = adv_emb.detach().cpu().numpy()
                    all_embeddings.append(batch_emb)
            else:
                # Regular evaluation
                inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1)
                    batch_scores_soft = probs[:, 2].cpu().numpy()
                    
                scores_soft.extend(batch_scores_soft)
                scores_hard.extend(batch_scores_soft)  # Same as soft
                is_adv_list.extend([0] * len(batch_texts))    
                
                if save_embeddings:
                    with torch.no_grad():
                        emb = self.model.deberta.embeddings.word_embeddings(inputs['input_ids'])
                    batch_emb = emb.detach().cpu().numpy()
                    all_embeddings.append(batch_emb)
            
            labels_list.extend(batch_labels)
        
        self.logger.info(f"Completed {eval_type} evaluation")
        if save_embeddings:
            all_embeddings = np.concatenate(all_embeddings, axis=0)
        return np.array(scores_soft), np.array(scores_hard), np.array(labels_list), np.array(is_adv_list), all_embeddings
    
    def generate_adversarial_batch(self, texts, labels, attack=None, attack_steps=20):
        """Generate adversarial examples by modifying one random token per positive input"""
        self.model.eval()
        
        # Tokenize all inputs
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Only attack positive examples
        positive_class_indices = torch.where(torch.tensor(labels) == 2)[0]
        if len(positive_class_indices) == 0:
            # Return original embeddings if no positives in batch
            with torch.no_grad():
                raw_emb = self.model.deberta.embeddings.word_embeddings(input_ids)
            return raw_emb, attention_mask, input_ids
        
        # Get embeddings for all examples
        with torch.no_grad():
            raw_emb = self.model.deberta.embeddings.word_embeddings(input_ids)
        
        # Select one random token position to modify per example
        seq_len = raw_emb.shape[1]
        token_positions = torch.randint(0, 10, (len(positive_class_indices),))
        
        final_emb, attention_mask = attack(raw_emb, attention_mask, token_positions, positive_class_indices, attack_steps)
        
        if type(attack).__name__ == "GCG":
            input_ids_hard = input_ids.clone()
            for i, idx in enumerate(positive_class_indices):
                for pos in range(10):
                    # Get adversarial embedding for this token
                    adv_token_emb = final_emb[idx, pos]
                    
                    # Find closest hard token ID
                    _, closest_token_id = self.distance_to_closest_hard_token(adv_token_emb.unsqueeze(0))
                    closest_token_id = closest_token_id.squeeze().item()
                    
                    # Update input_ids_hard
                    input_ids_hard[idx, pos] = closest_token_id
            assert final_emb == self.model.deberta.embeddings.word_embeddings(input_ids_hard)
            self.logger.info(f"GCG final_emb == input_ids_hard confimed.")
            
        else:
            # Replace attacked positions with closest hard tokens
            input_ids_hard = input_ids.clone()
            for i, (idx, pos) in enumerate(zip(positive_class_indices, token_positions)):
                # Get adversarial embedding for this token
                adv_token_emb = final_emb[idx, pos]
                
                # Find closest hard token ID
                _, closest_token_id = self.distance_to_closest_hard_token(adv_token_emb.unsqueeze(0))
                closest_token_id = closest_token_id.squeeze().item()
                
                # Update input_ids_hard
                input_ids_hard[idx, pos] = closest_token_id
        
        return final_emb, attention_mask, input_ids_hard
    
    def train_epoch(self, train_dataset, optimizer, batch_size=16, attack_steps=20, random_steps_per_batch=True):
        """Train for one epoch with adversarial examples"""
        self.model.train()
        total_loss = 0
        texts = train_dataset['text']
        labels = train_dataset['label']
        
        self.logger.info(f"Starting training epoch with {len(texts)} samples")
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Training"):
            batch_texts = texts[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            # Generate adversarial embeddings
            if random_steps_per_batch:
                # roughly 20% no attack, 20% full steps, rest in between.
                num_steps = np.clip(np.random.randint(low=-attack_steps//3, high=attack_steps+attack_steps//3), 0, attack_steps)
            adv_emb, attention_mask, adv_hard = self.generate_adversarial_batch(
                batch_texts, batch_labels, self.attack, num_steps
            )
            
            # Forward pass with aligned labels
            optimizer.zero_grad()
            outputs = self.model.deberta.encoder(adv_emb, attention_mask)[0]
            logits = self.model.classifier(self.model.pooler(outputs))
            loss = torch.nn.functional.cross_entropy(
                logits, 
                self._align_labels(batch_labels)
            )
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / (len(texts) / batch_size)
        self.logger.info(f"Completed training epoch - Average loss: {avg_loss:.4f}")
        return avg_loss
    
    def adversarial_train(self, train_dataset, test_dataset, epochs=5, lr=5e-5,
                         batch_size=16, attack_steps=20):
        """Full adversarial training loop with comprehensive evaluation"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        test_dataset_gcg = test_dataset.select(range(2*batch_size))
        eval_config = self.config.get('evaluation', {
                'run_benign': True,
                'run_train_attack': True,
                'run_strong_attack': True,
                'run_gcg': True
            })
        
        # Log training configuration
        self.logger.info("\n" + "="*50)
        self.logger.info("Starting Adversarial Training")
        self.logger.info(f"Training samples: {len(train_dataset)}")
        self.logger.info(f"Test samples: {len(test_dataset)}")
        self.logger.info(f"Epochs: {epochs}")
        self.logger.info(f"Batch size: {batch_size}")
        self.logger.info(f"Learning rate: {lr}")
        self.logger.info(f"Attack steps: {attack_steps}")
        self.logger.info("="*50 + "\n")
        
        # Initial evaluation
        self.logger.info("\nInitial Evaluation:")
        
        # Evaluate on different conditions
        if eval_config['run_benign']:
            benign_soft, benign_hard, benign_labels, _, _  = self.evaluate(test_dataset, batch_size, self.attack, attack_steps=0)
            benign_auc = roc_auc_score(benign_labels, benign_soft)
            benign_hard_auc = roc_auc_score(benign_labels, benign_hard)
            self.logger.info(f"Benign AUC: {benign_auc:.4f}")
            self.logger.info(f"Benign AUC (hard): {benign_hard_auc:.4f}")
        if eval_config['run_train_attack']:
            train_soft, train_hard, train_labels, _, _ = self.evaluate(test_dataset, batch_size, self.attack, attack_steps=attack_steps)
            train_attack_auc = roc_auc_score(train_labels, train_soft)
            train_attack_hard_auc = roc_auc_score(train_labels, train_hard)
            self.logger.info(f"Training Attack AUC: {train_attack_auc:.4f}")
            self.logger.info(f"Training Attack AUC (hard): {train_attack_hard_auc:.4f}")
        if eval_config['run_strong_attack']:
            strong_soft, strong_hard, strong_labels, _, _ = self.evaluate(test_dataset, batch_size, self.attack, attack_steps=attack_steps*2)
            strong_attack_auc = roc_auc_score(strong_labels, strong_soft)
            strong_attack_hard_auc = roc_auc_score(strong_labels, strong_hard)
            self.logger.info(f"Strong Attack AUC: {strong_attack_auc:.4f}")
            self.logger.info(f"Strong Attack AUC (hard): {strong_attack_hard_auc:.4f}")
        if eval_config['run_gcg']:
            gcg_soft, gcg_hard, gcg_labels, _, _ = self.evaluate(test_dataset_gcg, batch_size, self.gcg_attack, attack_steps=100)
            gcg_benign_soft, _, gcg_benign_labels, _, _ = self.evaluate(test_dataset_gcg, batch_size, self.attack, attack_steps=0)
        
            gcg_benign_auc = roc_auc_score(gcg_benign_labels, gcg_benign_soft)
            gcg_soft_auc = roc_auc_score(gcg_labels, gcg_soft)
            gcg_hard_auc = roc_auc_score(gcg_labels, gcg_hard)
            self.logger.info(f"GCG ({len(test_dataset_gcg)} samples)")         
            self.logger.info(f"GCG AUC: {gcg_soft_auc:.4f}")        
            self.logger.info(f"GCG AUC (hard ==): {gcg_hard_auc:.4f}")
            self.logger.info(f"benign AUC GCG subset: {gcg_benign_auc:.4f}")
        
        # Store metrics for final report
        metrics_history = {
            'epoch': [],
            'train_loss': [],
            'benign_auc': [],
            'train_attack_auc': [],
            'strong_attack_auc': [],
            'train_attack_hard_auc': [],
            'strong_attack_hard_auc': []
        }
        
        for dset_epoch in range(epochs):
            
            self.logger.info(f"\nEpoch {dset_epoch}/{epochs}")
            
            shuffled_train = train_dataset.shuffle()
            subset_size = 1024
            num_subepochs = len(train_dataset)//subset_size
            subsets = [
                shuffled_train.select(range(i*subset_size, (i+1)*subset_size))
                for i in range(num_subepochs)
            ]
            
            for subset_idx, subset in enumerate(subsets):
                self.logger.info(f"\nTraining Subset {subset_idx}/{num_subepochs}")
            
                
                # Train subepoch
                train_loss = self.train_epoch(subset, optimizer, batch_size, attack_steps)
                self.logger.info(f"Train Loss: {train_loss:.4f}")
                metrics_history['train_loss'].append(train_loss)
                
                #  Evaluate on different conditions
                if eval_config['run_benign']:
                    benign_soft, benign_hard, benign_labels, _, _  = self.evaluate(test_dataset, batch_size, self.attack, attack_steps=0)
                
                    benign_auc = roc_auc_score(benign_labels, benign_soft)
                    benign_hard_auc = roc_auc_score(benign_labels, benign_hard)
                
                    self.logger.info(f"Benign AUC: {benign_auc:.4f}")
                    self.logger.info(f"Benign AUC (hard): {benign_hard_auc:.4f}")
                    metrics_history['benign_auc'].append(benign_auc)
                if eval_config['run_train_attack']:
                    train_soft, train_hard, train_labels, _, _ = self.evaluate(test_dataset, batch_size, self.attack, attack_steps=attack_steps)
                
                    train_attack_auc = roc_auc_score(train_labels, train_soft)
                    train_attack_hard_auc = roc_auc_score(train_labels, train_hard)
                
                    self.logger.info(f"Training Attack AUC: {train_attack_auc:.4f}")
                    self.logger.info(f"Training Attack AUC (hard): {train_attack_hard_auc:.4f}")
                    metrics_history['train_attack_auc'].append(train_attack_auc)
                    metrics_history['train_attack_hard_auc'].append(train_attack_hard_auc)
                if eval_config['run_strong_attack']:
                    strong_soft, strong_hard, strong_labels, _, _ = self.evaluate(test_dataset, batch_size, self.attack, attack_steps=attack_steps*2)
                
                    strong_attack_auc = roc_auc_score(strong_labels, strong_soft)
                    strong_attack_hard_auc = roc_auc_score(strong_labels, strong_hard)
                
                    self.logger.info(f"Strong Attack AUC: {strong_attack_auc:.4f}")
                    self.logger.info(f"Strong Attack AUC (hard): {strong_attack_hard_auc:.4f}")
                    metrics_history['strong_attack_auc'].append(strong_attack_auc)
                    metrics_history['strong_attack_hard_auc'].append(strong_attack_hard_auc)
                if eval_config['run_gcg']:
                    gcg_soft, gcg_hard, gcg_labels, _, _ = self.evaluate(test_dataset_gcg, batch_size, self.gcg_attack, attack_steps=100)
                    gcg_benign_soft, _, gcg_benign_labels, _, _ = self.evaluate(test_dataset_gcg, batch_size, self.attack, attack_steps=0)
                
                    gcg_benign_auc = roc_auc_score(gcg_benign_labels, gcg_benign_soft)
                    gcg_soft_auc = roc_auc_score(gcg_labels, gcg_soft)
                    gcg_hard_auc = roc_auc_score(gcg_labels, gcg_hard)
                    
                    self.logger.info(f"GCG ({len(test_dataset_gcg)} samples)")         
                    self.logger.info(f"GCG AUC: {gcg_soft_auc:.4f}")        
                    self.logger.info(f"GCG AUC (hard ==): {gcg_hard_auc:.4f}")
                    self.logger.info(f"benign AUC GCG subset: {gcg_benign_auc:.4f}")
                
        
        # Final comprehensive evaluation
        self.logger.info("\nFinal Evaluation:")
        # Evaluate on different conditions
        if eval_config['run_benign']:
            benign_soft, benign_hard, benign_labels, _, _  = self.evaluate(test_dataset, batch_size, self.attack, attack_steps=0)
            benign_auc = roc_auc_score(benign_labels, benign_soft)
            benign_hard_auc = roc_auc_score(benign_labels, benign_hard)
            self.logger.info(f"Benign AUC: {benign_auc:.4f}")
            self.logger.info(f"Benign AUC (hard): {benign_hard_auc:.4f}")
        if eval_config['run_train_attack']:
            train_soft, train_hard, train_labels, _, _ = self.evaluate(test_dataset, batch_size, self.attack, attack_steps=attack_steps)
            train_attack_auc = roc_auc_score(train_labels, train_soft)
            train_attack_hard_auc = roc_auc_score(train_labels, train_hard)
            self.logger.info(f"Training Attack AUC: {train_attack_auc:.4f}")
            self.logger.info(f"Training Attack AUC (hard): {train_attack_hard_auc:.4f}")
        if eval_config['run_strong_attack']:
            strong_soft, strong_hard, strong_labels, _, _ = self.evaluate(test_dataset, batch_size, self.attack, attack_steps=attack_steps*2)
            strong_attack_auc = roc_auc_score(strong_labels, strong_soft)
            strong_attack_hard_auc = roc_auc_score(strong_labels, strong_hard)
            self.logger.info(f"Strong Attack AUC: {strong_attack_auc:.4f}")
            self.logger.info(f"Strong Attack AUC (hard): {strong_attack_hard_auc:.4f}")
        if eval_config['run_gcg']:
            gcg_soft, gcg_hard, gcg_labels, _, _ = self.evaluate(test_dataset_gcg, batch_size, self.gcg_attack, attack_steps=100)
            gcg_benign_soft, _, gcg_benign_labels, _, _ = self.evaluate(test_dataset_gcg, batch_size, self.attack, attack_steps=0)
        
            gcg_benign_auc = roc_auc_score(gcg_benign_labels, gcg_benign_soft)
            gcg_soft_auc = roc_auc_score(gcg_labels, gcg_soft)
            gcg_hard_auc = roc_auc_score(gcg_labels, gcg_hard)
            self.logger.info(f"GCG ({len(test_dataset_gcg)} samples)")         
            self.logger.info(f"GCG AUC: {gcg_soft_auc:.4f}")        
            self.logger.info(f"GCG AUC (hard ==): {gcg_hard_auc:.4f}")
            self.logger.info(f"benign AUC GCG subset: {gcg_benign_auc:.4f}")
        
        # Log metrics summary
        self.logger.info("\nTraining Summary:")
        df_metrics = pandas.DataFrame(metrics_history)
        self.logger.info("\nMetrics History:\n" + str(df_metrics))
        
        # Return metrics for potential further analysis
        return metrics_history
        
        
    #Tests    
    
    def test_distance_calculation(self):
        test_emb = self.hard_tokens[42]  # Pick a known token embedding
        distances, ids = self.distance_to_closest_hard_token(test_emb)
        assert ids.item() == 42, "Closest token should be itself!"
        assert torch.isclose(distances, torch.tensor(0.0)), "Distance to self should be zero!"
        
if __name__ == "__main__":
    
    # Load config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Configure device with fallback
    device = config['model'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configure timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"adv_prompt_guard_{timestamp}.log"
    
    # Load model and tokenizer
    prompt_injection_model_name = 'meta-llama/Prompt-Guard-86M'
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    model = AutoModelForSequenceClassification.from_pretrained(config['model']['name']).to(device)
    
    # Load and preprocess datasets with correct label mapping
    dataset = load_dataset("parquet", data_files={
        'train': config['datasets']['train_path'],
        'test': config['datasets']['test_path']
    })
    
    # Dataset sampling
    train_range = config['datasets']['sample_size']['train']
    test_range = config['datasets']['sample_size']['test']
    
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    if train_range is not None:
        train_dataset = train_dataset.select(range(train_range))
    if test_range is not None:
        test_dataset = test_dataset.select(range(test_range))
    
    # Map labels
    def map_labels(example):
        example['label'] = 0 if int(example['label']) == 0 else 2
        return example
    
    train_dataset = train_dataset.map(map_labels)
    test_dataset = test_dataset.map(map_labels)
        
    # Initialize attack
    attack_module = importlib.import_module('attacks')
    strategy_class = getattr(attack_module, config['attack']['strategy'])
    attack = strategy_class(
        model=model,
        tokenizer=tokenizer,
        device=device,
        **config['attack'].get('params', {})
    )
    
    gcg_class = getattr(attack_module, 'GCG')
    gcg_attack = gcg_class(
        model=model,
        tokenizer=tokenizer,
        device=device,
        **config['attack'].get('params', {})
    )
    
    # Initialize trainer
    trainer = AdvPromptGuardTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        log_file=log_file,
        attack=attack,
        gcg_attack=gcg_attack,
        config=config
    )
    
    # Run training
    metrics = trainer.adversarial_train(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        epochs=config['training']['epochs'],
        lr=config['training']['lr'],
        batch_size=config['training']['batch_size'],
        attack_steps=config['attack'].get('attack_steps', 20),
    )

    from saving_utils import save_model_and_data, upload_to_hf_hub
    save_model_and_data(trainer, test_dataset, config['attack']['attack_steps'])
    upload_to_hf_hub("your-username/your-model-name", "your-hf-token")
