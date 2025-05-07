import torch
import logging
class SoftTokenAttack:
    def __init__(self, model, tokenizer, device, logger=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        self.logger = logger or logging.getLogger(__name__)
                
        # adjust this if not using prompt-guard or other deberta model  
        self.hard_tokens = model.deberta.embeddings.word_embeddings.weight
        self.ites_diameter = self.hard_tokens.max() - self.hard_tokens.min()
        
    def compute_perturbation(self, grad, probs, step):
        raise NotImplementedError
        
    def apply_projection(self, adv_emb, raw_emb, pos_indices, token_positions, step):
        return adv_emb
    
    def __call__(self, raw_emb, attention_mask, token_positions, attacked_indices):
        return NotImplementedError

class OneTokenBenignGradAttack(SoftTokenAttack):
    def __init__(self, model, tokenizer, device, **kwargs):
        super().__init__(model, tokenizer, device)      

    
    def __call__(self, raw_emb, attention_mask, token_positions, attacked_indices, attack_steps):
        adv_emb = raw_emb.clone().detach().requires_grad_(True)
        for step in range(attack_steps):
            if adv_emb.grad is not None:
                adv_emb.grad.zero_()
            
            # Forward pass still on all samples
            modified_emb = raw_emb.clone()
            for i, position in enumerate(token_positions):
                modified_emb[attacked_indices[i], position] = adv_emb[attacked_indices[i], position]
            
            norm_emb = self.model.deberta.embeddings.LayerNorm(modified_emb)
            outputs = self.model.deberta.encoder(norm_emb, attention_mask)[0]
            logits = self.model.classifier(self.model.pooler(outputs))
            probs = torch.softmax(logits, dim=-1)
            
            # Maximize benign class (0) probability for jailbreak examples
            loss = -torch.log(probs[attacked_indices, 0]).mean()
            loss.backward()
                        
            adv_emb = self.compute_perturbation(adv_emb, raw_emb, loss, probs, step, attack_steps, attacked_indices, token_positions)
            
        # Create final embeddings with adversarial modifications
        final_emb = raw_emb.clone()
        for i, pos in enumerate(token_positions):
            final_emb[attacked_indices[i], pos] = adv_emb[attacked_indices[i], pos].detach()
            
        return final_emb, attention_mask
    
    def compute_perturbation(self, adv_emb, raw_emb, loss, probs, step, total_steps, attacked_indices, token_positions):
        raise NotImplementedError
    
class OneTokenLocalL2(OneTokenBenignGradAttack):
    def __init__(self, model, tokenizer, device, lr_l2=0.08, lr_linf=0.006, pen_l2=0.4, step_size_decay=0.94, **kwargs):
        super().__init__(model, tokenizer, device)
        
        self.lr_l2 = lr_l2
        self.lr_linf = lr_linf
        self.pen_l2 = pen_l2
        self.step_size_decay = step_size_decay
    
    # One should be able to inherit form OneTokenLocalL2 and only change the following to modify the attack.
    def compute_perturbation(self, adv_emb, raw_emb, loss, probs, step, total_steps, attacked_indices, token_positions):
        assert adv_emb.grad is not None
        with torch.no_grad():
            for i, pos in enumerate(token_positions): #could be parallelized, but should not take much time anyway
                idx = attacked_indices[i]
                grad = adv_emb.grad[idx, pos]
                perturbation = (
                    -grad * (self.lr_l2 * self.ites_diameter / (grad.abs().max() + 1e-8))  # L2++
                    -grad.sign() * self.lr_linf * self.ites_diameter * probs[idx, 2].item()  # Linf
                    -((adv_emb[idx, pos] - raw_emb[idx, pos])/self.ites_diameter * self.pen_l2 * probs[idx, 0].item())  # L2 penalty
                )
                adv_emb[idx, pos] += (self.step_size_decay**step) * perturbation
        return adv_emb
        
    
class OneTokenGlobalL2(OneTokenBenignGradAttack):
    def __init__(self, model, tokenizer, device, lr_l2=0.08, lr_linf=0.006, pen_l2=0.2, step_size_decay=0.94, radius_overlap=1.1, **kwargs):
        super().__init__(model, tokenizer, device)
        
        self.lr_l2 = lr_l2
        self.lr_linf = lr_linf
        self.pen_l2 = pen_l2
        self.step_size_decay = step_size_decay
        self.radius_overlap = radius_overlap
        
        self.R = self.compute_radius(sample_size=7000) 
        self.logger.info(f"[Attack] Initialized with radius R={self.R:.4f}")
        
    def compute_radius(self, sample_size=1000):
        """Approximate max token distance using a subset of tokens"""
        if len(self.hard_tokens) > 10_000:  # Only sample if vocab is large
            indices = torch.randint(0, len(self.hard_tokens), (sample_size,))
            sampled_tokens = self.hard_tokens[indices]
        else:
            sampled_tokens = self.hard_tokens
        
        # Compute on CPU to save GPU memory
        sampled_tokens_cpu = sampled_tokens.to("cpu")
        pairwise_dist = torch.cdist(sampled_tokens_cpu, sampled_tokens_cpu, p=2)
        # The code you provided is not complete and does not perform any specific action. It seems
        # like you have defined a variable `max_dist` but have not assigned any value to it or used it
        # in any operation. If you provide more context or complete the code snippet, I can help you
        # understand its functionality.
        # The above code is not complete and does not perform any specific task. It seems like a
        # variable `max_dist` is being declared but not assigned any value or used in any operation.
        max_dist = pairwise_dist.max()
        
        return (max_dist * self.radius_overlap / 2).to(self.hard_tokens.device)
    
    # One should be able to inherit form OneTokenLocalL2 and only change the following to modify the attack.
    def compute_perturbation(self, adv_emb, raw_emb, loss, probs, step, total_steps, attacked_indices, token_positions):
        assert adv_emb.grad is not None
        with torch.no_grad():
            if step < total_steps//3:
                for i, pos in enumerate(token_positions): #could be parallelized, but should not take much time anyway
                    idx = attacked_indices[i]
                    grad = adv_emb.grad[idx, pos]
                    perturbation = (
                        -grad * (self.lr_l2 * self.ites_diameter / (grad.abs().max() + 1e-8))  # L2++
                        -grad.sign() * self.lr_linf * self.ites_diameter * probs[idx, 2].item()  # Linf
                    )
                    adv_emb[idx, pos] += (self.step_size_decay**step) * perturbation
                    
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        dists = torch.norm(adv_emb[idx, pos].half() - self.hard_tokens.half(), dim=1)
                    v0 = torch.argmin(dists.float())
                    
                    if dists[v0] > self.R:
                        vec_to_hard_token = self.hard_tokens[v0] - adv_emb[idx, pos]
                        adv_emb[idx, pos] += vec_to_hard_token * (1 - self.R/torch.norm(vec_to_hard_token))
                        
            elif step < 2*total_steps//3:
                for i, pos in enumerate(token_positions): #could be parallelized, but should not take much time anyway
                    idx = attacked_indices[i]
                    grad = adv_emb.grad[idx, pos]
                    perturbation = (
                        -grad * (self.lr_l2 * self.ites_diameter / (grad.abs().max() + 1e-8))  # L2++
                        -grad.sign() * self.lr_linf * self.ites_diameter * probs[idx, 2].item()  # Linf
                    )
                    adv_emb[idx, pos] += (self.step_size_decay**step) * perturbation
                    
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        dists = torch.norm(adv_emb[idx, pos].half() - self.hard_tokens.half(), dim=1)
                    v0 = torch.argmin(dists.float())
                    
                    vec_to_hard_token = self.hard_tokens[v0] - adv_emb[idx, pos]
                    adv_emb[idx, pos] += (self.step_size_decay**step) * vec_to_hard_token * self.pen_l2 * probs[idx, 0].item()
                    vec_to_hard_token = self.hard_tokens[v0] - adv_emb[idx, pos]
                    if dists[v0] > self.R:
                        vec_to_hard_token = self.hard_tokens[v0] - adv_emb[idx, pos]
                        adv_emb[idx, pos] += vec_to_hard_token * (1 - self.R/torch.norm(vec_to_hard_token))
                        
            else:
                for i, pos in enumerate(token_positions): #could be parallelized, but should not take much time anyway
                    idx = attacked_indices[i]
                    grad = adv_emb.grad[idx, pos]
                    perturbation = (
                        -grad * (self.lr_l2 * self.ites_diameter / (grad.abs().max() + 1e-8))  # L2++
                        -grad.sign() * self.lr_linf * self.ites_diameter * probs[idx, 2].item()  # Linf
                    )
                    adv_emb[idx, pos] += (self.step_size_decay**step) * perturbation
                    
                    
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        dists = torch.norm(adv_emb[idx, pos].half() - self.hard_tokens.half(), dim=1)
                    v0 = torch.argmin(dists.float())
                    
                    vec_to_hard_token = self.hard_tokens[v0] - adv_emb[idx, pos]
                    adv_emb[idx, pos] += (self.step_size_decay**step) * vec_to_hard_token * self.pen_l2 * probs[idx, 0].item()
                    vec_to_hard_token = self.hard_tokens[v0] - adv_emb[idx, pos]
                    R_step = self.R*(1. - 0.3*(step+10)/(total_steps+9))
                    if torch.norm(vec_to_hard_token) > R_step:
                        adv_emb[idx, pos] += vec_to_hard_token * (1 - R_step/torch.norm(vec_to_hard_token))
                    #print(f'{torch.norm(self.hard_tokens[v0] - adv_emb[idx, pos]).item():.4f}, {self.R:.4f}')

        #print(f'{step:02d} {probs[:,2].mean().item():.4f},  {torch.norm(adv_emb-raw_emb).mean().item():.4f}, {self.R:.4f}')
        return adv_emb



class MultiStepAttackStrategy(SoftTokenAttack):
    def __init__(self, model, tokenizer, device, **kwargs):
        super().__init__(model, tokenizer, device)
        self.R = self._compute_radius()
        
    def _compute_radius(self):
        pairwise_dist = torch.cdist(self.hard_tokens, self.hard_tokens, p=2)
        return pairwise_dist.max() / 1.9
        
    def compute_perturbation(self, grad, probs, step):
        if step < 8:
            return 0.1 * grad + 0.01 * grad.sign()
        elif step < 16:
            return 0.1 * grad + 0.01 * grad.sign() - 0.05 * (self.current_emb - self.raw_emb)
        else:
            return 0.1 * grad
        
    def apply_projection(self, adv_emb, raw_emb, pos_indices, token_positions, step):
        if step >= 8:
            with torch.no_grad():
                for i, pos in enumerate(token_positions):
                    idx = pos_indices[i]
                    current_emb = adv_emb[idx, pos]
                    dists = torch.norm(current_emb - self.hard_tokens, dim=1)
                    v0 = torch.argmin(dists)
                    if dists[v0] > self.R:
                        direction = self.hard_tokens[v0] - current_emb
                        adv_emb[idx, pos] += direction * (dists[v0] - self.R)/dists[v0]
        return adv_emb
    
    
class GCG(SoftTokenAttack):
    def __init__(self, model, tokenizer, device, **kwargs):
        super().__init__(model, tokenizer, device)

    
    def __call__(self, raw_emb, attention_mask, token_positions, attacked_indices, attack_steps=100, token_possible_positions=range(10), topk=32):
        adv_emb = raw_emb.clone().detach().requires_grad_(True)
        for step in range(attack_steps):
            if adv_emb.grad is not None:
                adv_emb.grad.zero_()
            
            # Forward pass still on all samples
            norm_emb = self.model.deberta.embeddings.LayerNorm(adv_emb)
            outputs = self.model.deberta.encoder(norm_emb, attention_mask)[0]
            logits = self.model.classifier(self.model.pooler(outputs))
            probs = torch.softmax(logits, dim=-1)
            
            # Maximize benign class (0) probability for jailbreak examples
            loss = -torch.log(probs[attacked_indices, 0]).mean()
            loss.backward()
            
            # adv_emb: B, T, E
            # word_embeddings.weight: V, E
            hard_token_gradients = torch.einsum('bte,ve->btv', -adv_emb.grad, self.hard_tokens)
            self.model.zero_grad()
            
            top_hard_token_gradients = torch.topk(hard_token_gradients, topk, dim=-1).indices # same batch size for forward pass per sample
            for j in attacked_indices:
                candidates = [adv_emb[j].clone()]
                for b in range(raw_emb.shape[0]-1): #same batch size as outside for forward pass
                    replaced_pos = token_possible_positions[torch.randint(len(token_possible_positions))]
                    choice_from_topk = torch.randint(topk)
                    replacement_token_id = top_hard_token_gradients[j, replaced_pos, choice_from_topk]
                    replacement_token = self.hard_tokens[replacement_token_id]
                    candidate = adv_emb[j].clone()
                    candidate[replaced_pos] = replacement_token
                    candidates.append[candidate]
                candidates = torch.stack(candidates)
                attention_mask_candidates = torch.stack(len(candidates)*[attention_mask[j]])
                adv_emb[j] = self.best_candidate(candidates, attention_mask_candidates)
        
            
        return adv_emb, attention_mask
    
    def best_candidate(self, candidate_embs, attention_mask):
        """Evaluate candidate replacements for one sequence and return best. I.e. batch dim not the same thing as elsewhere."""
        with torch.no_grad():
            outputs = self.model.deberta.encoder(candidate_embs, attention_mask)[0]
            logits = self.model.classifier(self.model.pooler(outputs))
            probs = torch.softmax(logits, dim=-1)
            
        return candidate_embs[torch.argmax(probs[:,0])]
    