import json
import torch
import torch.distributed as dist
import torch.nn.functional as F
from typing import List, Union, Optional, Tuple, Mapping, Dict


def save_json_to_file(objects: Union[List, dict], path: str, line_by_line: bool = False):
    if line_by_line:
        assert isinstance(objects, list), 'Only list can be saved in line by line format'

    with open(path, 'w', encoding='utf-8') as writer:
        if not line_by_line:
            json.dump(objects, writer, ensure_ascii=False, indent=4, separators=(',', ':'))
        else:
            for obj in objects:
                writer.write(json.dumps(obj, ensure_ascii=False, separators=(',', ':')))
                writer.write('\n')

def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda(non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return tuple([_move_to_cuda(x) for x in maybe_tensor])
        elif isinstance(maybe_tensor, Mapping):
            return type(maybe_tensor)({k: _move_to_cuda(v) for k, v in maybe_tensor.items()})
        else:
            return maybe_tensor

    return _move_to_cuda(sample)


def dist_gather_tensor(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if t is None:
        return None

    t = t.contiguous()
    all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(all_tensors, t)

    all_tensors[dist.get_rank()] = t
    all_tensors = torch.cat(all_tensors, dim=0)
    return all_tensors


@torch.no_grad()
def select_grouped_indices(scores: torch.Tensor,
                           group_size: int,
                           start: int = 0) -> torch.Tensor:
    assert len(scores.shape) == 2
    batch_size = scores.shape[0]
    #assert batch_size * group_size <= scores.shape[1]

    indices = torch.arange(0, group_size, dtype=torch.long)
    indices = indices.repeat(batch_size, 1)
    indices += torch.arange(0, batch_size, dtype=torch.long).unsqueeze(-1) * group_size
    indices += start

    return indices.to(scores.device)

def _sample_from_matrix(
    matrix: torch.Tensor,
    num_samples: int,
    strategy: Optional[str]
) -> torch.Tensor:
    """
    Internal helper to sample from a square score matrix (kq, qq, or kk).

    Args:
        matrix: The square score matrix to sample from.
        num_samples: The number of items to sample if a strategy is used.
        strategy: 'hard', 'random', or None. If None, all items are returned.
    """
    # Always mask the diagonal to prevent sampling self-similarity
    matrix.fill_diagonal_(float('-inf'))

    if strategy == 'None' or strategy is None:
        # If strategy is None, use all available pairs.
        return matrix, None
    
    batch_size = matrix.shape[0]
    num_available = batch_size - 1
    
    effective_num_samples = min(num_samples, num_available) if num_samples >= 0 else num_available
    
    if effective_num_samples <= 0:
        return torch.empty((batch_size, 0), device=matrix.device, dtype=matrix.dtype), None

    if strategy == 'hard':
        sampled_scores, sample_indices = torch.topk(matrix, k=effective_num_samples, dim=1)
        return sampled_scores, sample_indices
    
    elif strategy == 'random':
        weights = (matrix != float('-inf')).float()
        random_indices = torch.multinomial(
            weights, num_samples=effective_num_samples, replacement=False
        )
        return torch.gather(matrix, 1, random_indices), random_indices
    
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}. Choose 'hard', 'random', or None.")

def enhanced_hard_negative_mining(
    query: torch.Tensor,
    key: torch.Tensor,
    local_qk_scores: torch.Tensor,
    qk_full: torch.Tensor,
    train_n_passages: int,
    n_hard_neg: int,
    constraint_type: str = "none",
    margin: float = 0.0,
    percentage: float = 1.0,
    extra_key: torch.Tensor = None,
    fn_kk_threshold: float = 0,
    embedding_dedup_threshold: float = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Enhanced hard negative mining with flexible constraint options.
    
    Args:
        query: Query embeddings (batch_size, dim)
        key: Key embeddings (batch_size * train_n_passages, dim)
        local_qk_scores: Local query-key scores (batch_size, train_n_passages)
        qk_full: Full query-key similarity matrix (batch_size, batch_size * train_n_passages)
        train_n_passages: Number of passages per query
        n_hard_neg: Number of hard negatives to mine
        constraint_type: Type of constraint to apply:
            - "none": No constraint (original behavior)
            - "margin": negative_score < max_positive_score - margin
            - "percentage": negative_score < max_positive_score * percentage
        margin: Margin value for margin-based constraint (default: 0.0)
        percentage: Percentage value for percentage-based constraint (default: 1.0, range: 0.0-1.0)
        extra_key: Additional keys for false negative filtering
        fn_kk_threshold: Threshold for false negative filtering
        embedding_dedup_threshold: Threshold for embedding deduplication
        
    Returns:
        hard_negative_scores: Scores of selected hard negatives (batch_size, n_hard_neg)
        hard_negative_indices: Indices of selected hard negatives (batch_size, n_hard_neg)
        masked_qk: The masked query-key matrix used for sampling (for random negatives)
    """
    batch_size = query.shape[0]
    
    # Create a list of `batch_size` blocks, each of shape (1, train_n_passages)
    local_passage_blocks = [
        torch.ones(1, train_n_passages, dtype=torch.bool, device=query.device)
    ] * batch_size    
    # Create the block-diagonal mask identifying the local passages
    local_passage_mask = torch.block_diag(*local_passage_blocks)
    masked_qk = qk_full.masked_fill(local_passage_mask, float('-inf'))

    # Apply false negative filtering if specified
    if fn_kk_threshold > 0 and extra_key is not None:
        num_positives = extra_key.shape[0] // batch_size
        extra_key = extra_key.view(num_positives, batch_size, query.shape[1])
        kk_full_all_positives = torch.einsum("pbd,nd->pbn", extra_key, key)
        local_passage_mask_expanded = local_passage_mask.unsqueeze(0).expand_as(kk_full_all_positives)
        kk_full_all_positives = kk_full_all_positives.masked_fill(local_passage_mask_expanded, float('-inf'))
        kk_full = kk_full_all_positives.max(dim=0).values
        false_negative_mask = (kk_full >= fn_kk_threshold)
        masked_qk = masked_qk.masked_fill(false_negative_mask, float('-inf'))

    # Apply embedding deduplication if specified
    if embedding_dedup_threshold > 0:
        kk_all = torch.mm(key, key.t())
        highly_similar_matrix = (kk_all > embedding_dedup_threshold).tril(diagonal=-1)
        is_embedding_duplicate_mask = torch.any(highly_similar_matrix, dim=0).unsqueeze(0).expand(batch_size, -1)
        masked_qk = masked_qk.masked_fill(is_embedding_duplicate_mask, float('-inf'))

    # Enhanced hard negative mining with flexible constraint options
    if constraint_type == "margin":
        # Option 1: negative_score < max_positive_score - margin
        max_positive_scores = local_qk_scores.max(dim=1, keepdim=True).values  # Shape: (batch_size, 1)
        threshold_scores = max_positive_scores - margin
        
        # Create a mask for scores that are below the threshold
        below_threshold_mask = masked_qk < threshold_scores  # Shape: (batch_size, num_keys)
        
        # Apply the mask: set scores above threshold to -inf
        masked_qk_constrained = masked_qk.masked_fill(~below_threshold_mask, float('-inf'))
        
        # Select hard negatives from the filtered pool
        hard_negative_scores, hard_negative_indices = torch.topk(masked_qk_constrained, k=n_hard_neg, dim=1)
    
    elif constraint_type == "percentage":
        # Option 2: negative_score < max_positive_score * percentage
        max_positive_scores = local_qk_scores.max(dim=1, keepdim=True).values  # Shape: (batch_size, 1)
        threshold_scores = max_positive_scores * percentage
        
        # Create a mask for scores that are below the threshold
        below_threshold_mask = masked_qk < threshold_scores  # Shape: (batch_size, num_keys)
        
        # Apply the mask: set scores above threshold to -inf
        masked_qk_constrained = masked_qk.masked_fill(~below_threshold_mask, float('-inf'))
        
        # Select hard negatives from the filtered pool
        hard_negative_scores, hard_negative_indices = torch.topk(masked_qk_constrained, k=n_hard_neg, dim=1)
    
    else:
        # No constraint (original behavior): select top scoring negatives
        hard_negative_scores, hard_negative_indices = torch.topk(masked_qk, k=n_hard_neg, dim=1)
    
    return hard_negative_scores, hard_negative_indices, masked_qk


def compute_regularization(qq, positive_docs, tau=0.5):
    unique_pos_docs, inverse_indices = torch.unique(positive_docs, return_inverse=True, dim=0)
    group_membership_matrix = F.one_hot(inverse_indices, num_classes=unique_pos_docs.shape[0]).to(qq.dtype) # group_membership_matrix[t, j] = 1 if query t belongs to product group j.
    group_sizes = group_membership_matrix.sum(dim=0, keepdim=True) # group_sizes[j] = number of queries in product group j.
    summed_sim_matrix = qq.T @ group_membership_matrix # summed_sim_matrix[i, j] = SUM_{t in T_j} [ sim(q_i, q_t) ]
    theta = summed_sim_matrix / group_sizes
    regularization = (1-theta)**tau 
    return regularization

def full_contrastive_scores_and_labels(
        query: torch.Tensor,
        key: torch.Tensor,
        extra_key: torch.Tensor = None,
        n_hard_neg: Optional[int] = None,
        n_rand_neg: Optional[int] = None,
        n_other_neg: Optional[int] = None,
        other_neg_sampling_strategy: Optional[str] = None,
        use_all_pairs: bool = True,
        add_qq_regularization: bool = False,
        fn_kk_threshold: float = 0,
        embedding_dedup_threshold: float = 0,
        hard_neg_constraint_type: str = "none",
        hard_neg_margin: float = 0.0,
        hard_neg_percentage: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, Optional[dict]]:
    assert key.shape[0] % query.shape[0] == 0, '{} % {} > 0'.format(key.shape[0], query.shape[0])
    batch_size, dim = query.shape
    train_n_passages = key.shape[0] // query.shape[0]
    labels = torch.arange(0, query.shape[0], dtype=torch.long, device=query.device)
    labels = (labels * train_n_passages)

    # batch_size x dim
    sliced_key = key.index_select(dim=0, index=labels) # positive docs
    assert query.shape[0] == sliced_key.shape[0]

    # Calculate Local Scores (query_i vs its own passage_block_i)
    query_reshaped = query.unsqueeze(1)
    key_reshaped = key.view(batch_size, train_n_passages, dim)
    local_qk_scores = torch.bmm(query_reshaped, key_reshaped.transpose(1, 2)).squeeze(1)
    
    # batch_size x (batch_size x n_psg)
    qk_full = torch.mm(query, key.t())

    # logging purpose
    #qq = torch.mm(query, query.t())
    #res = {"qq":qq.cpu(), "query": query.cpu(), "key": key.cpu(), "qk_full": qk_full.cpu(), 'extra_key': extra_key.cpu()}
    #torch.save(res, f"/data/data.pt")

    #kk_all = torch.mm(key, key.t()) # Shape: (num_keys, num_keys)
    hard_negative_indices = None
    if n_hard_neg and n_hard_neg > 0:
        # Use the enhanced hard negative mining function
        hard_negative_scores, hard_negative_indices, masked_qk = enhanced_hard_negative_mining(
            query=query,
            key=key,
            local_qk_scores=local_qk_scores,
            qk_full=qk_full,
            train_n_passages=train_n_passages,
            n_hard_neg=n_hard_neg,
            constraint_type=hard_neg_constraint_type,
            margin=hard_neg_margin,
            percentage=hard_neg_percentage,
            extra_key=extra_key,
            fn_kk_threshold=fn_kk_threshold,
            embedding_dedup_threshold=embedding_dedup_threshold
        )
        # Initialize with a default empty tensor. This will be used if n_rand_neg is 0.
        random_negative_scores = torch.empty((batch_size, 0), device=query.device, dtype=query.dtype)
        # Update the pool by masking out the hard negatives we just chose
        if n_rand_neg and n_rand_neg > 0:
            rs_pool = masked_qk.clone()
            rs_pool.scatter_(1, hard_negative_indices, float('-inf'))
            # Sample Random Negatives from the updated pool
            rs_mask = (rs_pool != float('-inf')).float()
            num_available = torch.sum(rs_mask, dim=1).long()
            if torch.any(num_available < n_rand_neg):
                pass
            random_negative_indices = torch.multinomial(
                rs_mask, num_samples=n_rand_neg, replacement=False # No replacement to get unique samples
            )
            # Gather the scores for the randomly sampled negatives
            random_negative_scores = torch.gather(qk_full, 1, random_negative_indices)
        # The new label for every example is 0.
        new_labels = torch.zeros(batch_size, dtype=torch.long, device=query.device)
        qk = torch.cat([local_qk_scores, hard_negative_scores, random_negative_scores], dim=-1)
        if not use_all_pairs:
            return qk, new_labels, None
    else:
        qk = qk_full
        new_labels = labels
        if not use_all_pairs:
            return qk, new_labels, None

    # batch_size x batch_size
    qq = torch.mm(query, query.t())
    kq = torch.mm(sliced_key, query.t())
    kk = torch.mm(sliced_key, sliced_key.t())

    kq_sampled, kq_indices = _sample_from_matrix(kq.clone(), n_other_neg or 0, other_neg_sampling_strategy)
    qq_sampled, qq_indices = _sample_from_matrix(qq.clone(), n_other_neg or 0, other_neg_sampling_strategy)
    kk_sampled, kk_indices = _sample_from_matrix(kk.clone(), n_other_neg or 0, other_neg_sampling_strategy)
    
    scores = torch.cat([qk, kq_sampled, qq_sampled, kk_sampled], dim=-1)

    sample_indices = {'qk': hard_negative_indices, 'kq': kq_indices, 'qq': qq_indices, 'kk': kk_indices}

    return scores, new_labels, sample_indices


def slice_batch_dict(batch_dict: Dict[str, torch.Tensor], prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in batch_dict.items() if k.startswith(prefix)}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, round_digits: int = 3):
        self.name = name
        self.round_digits = round_digits
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return '{}: {}'.format(self.name, round(self.avg, self.round_digits))


if __name__ == '__main__':
    torch.manual_seed(seed=0)
    query = torch.randn(4, 16)
    key = torch.randn(4 * 3, 16)
    extra_key = torch.randn(4 * 2, 16)
    scores, labels, indices = full_contrastive_scores_and_labels(query, key, extra_key, n_hard_neg=2, n_other_neg=3)
    print(scores.shape)

    print(labels)
