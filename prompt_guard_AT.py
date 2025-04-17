import logging
from datetime import datetime
import matplotlib.pyplot as plt
import pandas
import seaborn as sns
import time
import torch
import numpy as np

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
    def __init__(self, model, tokenizer, device='cpu', log_file='adv_prompt_guard.log'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
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
        
        # Log initialization info
        self.logger.info(f"Initializing AdvPromptGuardTrainer on device: {device}")
        self.logger.info(f"Model: {type(model).__name__}")
        self.logger.info(f"Tokenizer: {type(tokenizer).__name__}")
        
    def _align_labels(self, labels):
        """Map dataset labels (0,1) to model labels (0,2)"""
        return torch.tensor([0 if x == 0 else 2 for x in labels], device=self.device)
        
    def evaluate(self, dataset, batch_size=16, attack_steps=0, save_embeddings=False):
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
        scores = []
        is_adv = []
        all_embeddings = [] if save_embeddings else None
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            if attack_steps > 0:
                # Generate adversarial examples for positive samples
                adv_emb, attention_mask = self.generate_adversarial_batch(
                    batch_texts, batch_labels, attack_steps
                )
                # Directly use the adversarial embeddings
                outputs = self.model.deberta.encoder(adv_emb, attention_mask)[0]
                logits = self.model.classifier(self.model.pooler(outputs))
                probs = torch.softmax(logits, dim=-1)
                batch_scores = probs[:, 2].detach().cpu().numpy()  # Jailbreak class score
                batch_is_adv = [1 if label == 1 else 0 for label in batch_labels]
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
                    batch_scores = probs[:, 2].cpu().numpy()
                batch_is_adv = [0] * len(batch_texts)
                if save_embeddings:
                    with torch.no_grad():
                        emb = self.model.deberta.embeddings(inputs['input_ids'])
                    batch_emb = emb.detach().cpu().numpy()
                    all_embeddings.append(batch_emb)
            
            scores.extend(batch_scores)
            is_adv.extend(batch_is_adv)
        
        self.logger.info(f"Completed {eval_type} evaluation")
        if save_embeddings:
            all_embeddings = np.concatenate(all_embeddings, axis=0)
        return np.array(scores), np.array(labels), np.array(is_adv), all_embeddings
    
    def generate_adversarial_batch(self, texts, labels, attack_steps=20):
        """Generate adversarial examples by modifying one random token per positive input"""
        self.model.eval()
        
        # Tokenize all inputs
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Only attack positive examples
        pos_indices = torch.where(torch.tensor(labels) == 2)[0]
        if len(pos_indices) == 0:
            # Return original embeddings if no positives in batch
            with torch.no_grad():
                orig_emb = self.model.deberta.embeddings(input_ids)
            return orig_emb, attention_mask
        
        # Get embeddings for all examples
        with torch.no_grad():
            orig_emb = self.model.deberta.embeddings(input_ids)
        
        # Initialize adversarial embeddings (only for positive examples)
        adv_emb = orig_emb.clone().detach().requires_grad_(True)
        emb_range = orig_emb.max() - orig_emb.min()
        
        # Attack parameters
        lr_l2 = 0.08
        lr_linf = 0.006
        pen_l2 = 0.4
        lr_decay = 0.94
        
        # Select one random token position to modify per example
        seq_len = orig_emb.shape[1]
        token_positions = torch.randint(0, 10, (len(pos_indices),))
        
        for step in range(attack_steps):
            if adv_emb.grad is not None:
                adv_emb.grad.zero_()
            
            # Forward pass only on modified tokens
            modified_emb = orig_emb.clone()
            for i, pos in enumerate(token_positions):
                modified_emb[pos_indices[i], pos] = adv_emb[pos_indices[i], pos]
            
            outputs = self.model.deberta.encoder(modified_emb, attention_mask)[0]
            logits = self.model.classifier(self.model.pooler(outputs))
            probs = torch.softmax(logits, dim=-1)
            
            # Maximize benign class (0) probability for jailbreak examples
            loss = -torch.log(probs[pos_indices, 0]).mean()
            loss.backward()
            
            # Update only the selected token positions
            with torch.no_grad():
                for i, pos in enumerate(token_positions):
                    idx = pos_indices[i]
                    grad = adv_emb.grad[idx, pos]
                    perturbation = (
                        -grad * (lr_l2 * emb_range / (grad.abs().max() + 1e-8))  # L2
                        -grad.sign() * lr_linf * emb_range * probs[idx, 2].item()  # Linf
                        -((adv_emb[idx, pos] - orig_emb[idx, pos])/emb_range * pen_l2 * probs[idx, 0].item())  # L2 penalty
                    )
                    adv_emb[idx, pos] += (lr_decay**step) * perturbation

        # Create final embeddings with adversarial modifications
        final_emb = orig_emb.clone()
        for i, pos in enumerate(token_positions):
            final_emb[pos_indices[i], pos] = adv_emb[pos_indices[i], pos].detach()
        
        return final_emb, attention_mask
    
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
            adv_emb, attention_mask = self.generate_adversarial_batch(
                batch_texts, batch_labels, num_steps
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
        benign_scores, benign_labels, _, _ = self.evaluate(test_dataset, batch_size, attack_steps=0)
        train_attack_scores, train_attack_labels, _, _ = self.evaluate(test_dataset, batch_size, attack_steps=attack_steps)
        strong_attack_scores, strong_attack_labels, _, _ = self.evaluate(test_dataset, batch_size, attack_steps=attack_steps*2)
        
        benign_auc = roc_auc_score(benign_labels, benign_scores)
        train_attack_auc = roc_auc_score(train_attack_labels, train_attack_scores)
        strong_attack_auc = roc_auc_score(strong_attack_labels, strong_attack_scores)
        
        self.logger.info(f"Benign AUC: {benign_auc:.4f}")
        self.logger.info(f"Training Attack AUC: {train_attack_auc:.4f}")
        self.logger.info(f"Strong Attack AUC: {strong_attack_auc:.4f}")
        
        # Store metrics for final report
        metrics_history = {
            'epoch': [],
            'train_loss': [],
            'benign_auc': [],
            'train_attack_auc': [],
            'strong_attack_auc': []
        }
        
        for epoch in range(epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train epoch
            train_loss = self.train_epoch(train_dataset, optimizer, batch_size, attack_steps)
            self.logger.info(f"Train Loss: {train_loss:.4f}")
            
            # Evaluate on different conditions
            benign_scores, benign_labels, _, _ = self.evaluate(test_dataset, batch_size, attack_steps=0)
            train_attack_scores, train_attack_labels, _, _ = self.evaluate(test_dataset, batch_size, attack_steps=attack_steps)
            strong_attack_scores, strong_attack_labels, _, _ = self.evaluate(test_dataset, batch_size, attack_steps=attack_steps*2)
            
            benign_auc = roc_auc_score(benign_labels, benign_scores)
            train_attack_auc = roc_auc_score(train_attack_labels, train_attack_scores)
            strong_attack_auc = roc_auc_score(strong_attack_labels, strong_attack_scores)
            
            # Log epoch metrics
            self.logger.info(f"Benign AUC: {benign_auc:.4f}")
            self.logger.info(f"Training Attack AUC: {train_attack_auc:.4f}")
            self.logger.info(f"Strong Attack AUC: {strong_attack_auc:.4f}")
            
            # Store metrics
            metrics_history['epoch'].append(epoch + 1)
            metrics_history['train_loss'].append(train_loss)
            metrics_history['benign_auc'].append(benign_auc)
            metrics_history['train_attack_auc'].append(train_attack_auc)
            metrics_history['strong_attack_auc'].append(strong_attack_auc)
        
        # Final comprehensive evaluation
        self.logger.info("\nFinal Evaluation:")
        benign_scores, benign_labels, _ = self.evaluate(test_dataset, batch_size, attack_steps=0)
        train_attack_scores, train_attack_labels, _ = self.evaluate(test_dataset, batch_size, attack_steps=attack_steps)
        strong_attack_scores, strong_attack_labels, _ = self.evaluate(test_dataset, batch_size, attack_steps=attack_steps*2)
        
        benign_auc = roc_auc_score(benign_labels, benign_scores)
        train_attack_auc = roc_auc_score(train_attack_labels, train_attack_scores)
        strong_attack_auc = roc_auc_score(strong_attack_labels, strong_attack_scores)
        
        self.logger.info("\nFinal Performance:")
        self.logger.info(f"Benign AUC: {benign_auc:.4f}")
        self.logger.info(f"Training Attack AUC: {train_attack_auc:.4f}")
        self.logger.info(f"Strong Attack AUC: {strong_attack_auc:.4f}")
        
        # Log metrics summary
        self.logger.info("\nTraining Summary:")
        df_metrics = pandas.DataFrame(metrics_history)
        self.logger.info("\nMetrics History:\n" + str(df_metrics))
        
        # Return metrics for potential further analysis
        return metrics_history

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Configure timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"adv_prompt_guard_{timestamp}.log"
    
    # Load model and tokenizer
    prompt_injection_model_name = 'meta-llama/Prompt-Guard-86M'
    tokenizer = AutoTokenizer.from_pretrained(prompt_injection_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(prompt_injection_model_name).to(device)
    
    # Load and preprocess datasets with correct label mapping
    dataset = load_dataset("parquet", data_files={
        'train': 'synthetic-prompt-injections_train.parquet', 
        'test': 'synthetic-prompt-injections_test.parquet'
    })
    
    # Convert dataset labels: 0 remains 0 (benign), 1 becomes 2 (jailbreak)
    def map_labels(example):
        example['label'] = 0 if int(example['label']) == 0 else 2
        return example
    
    train_dataset = dataset['train'].select(range(10000)).map(map_labels)
    test_dataset = dataset['test'].select(range(100)).map(map_labels)
        
    # Initialize and run trainer
    trainer = AdvPromptGuardTrainer(model, tokenizer, device, log_file=log_file)
    metrics = trainer.adversarial_train(train_dataset, test_dataset, epochs=10, batch_size=16)
    
    from save_utils import save_model_and_data, upload_to_hf_hub
    save_model_and_data(trainer, test_dataset, attack_steps=20)
    upload_to_hf_hub("your-username/your-model-name", "your-hf-token")