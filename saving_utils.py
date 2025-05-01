# save_utils.py
import torch
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, login

def save_model_and_data(trainer, test_dataset, attack_steps, save_dir="output"):
    # Save model & tokenizer
    trainer.model.save_pretrained(f"{save_dir}/model")
    trainer.tokenizer.save_pretrained(f"{save_dir}/model")
    
    # Save evaluation data (texts + embeddings)
    def _save_eval_data(attack_steps_val, suffix):
        scores_soft, scores_hard, labels, is_adv, embeddings = trainer.evaluate(
            test_dataset, 
            attack_steps=attack_steps_val,
            save_embeddings=True
        )
        # Save texts and labels
        pd.DataFrame({
            'text': test_dataset['text'],
            'label': labels
        }).to_csv(f"{save_dir}/eval_{suffix}_texts.csv", index=False)
        
        # Save embeddings
        np.save(f"{save_dir}/eval_{suffix}_embeddings.npy", embeddings)

    _save_eval_data(0, "benign")
    _save_eval_data(attack_steps, "train_attack")
    _save_eval_data(attack_steps*2, "strong_attack")

def upload_to_hf_hub(repo_name, hf_token):
    login(token=hf_token)
    api = HfApi()
    api.create_repo(repo_id=repo_name, exist_ok=True)
    api.upload_folder(
        folder_path="output/model",
        repo_id=repo_name,
        repo_type="model"
    )