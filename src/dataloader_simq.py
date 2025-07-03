import os
import random
import json
from typing import Tuple, Dict, List, Optional, Iterator
from datasets import load_dataset, DatasetDict, Dataset
from transformers.file_utils import PaddingStrategy
from transformers import PreTrainedTokenizerFast, Trainer

from config import Arguments
from logger_config import logger
from .loader_utils import group_doc_ids


class RetrievalDataLoader:

    def __init__(self, args: Arguments, tokenizer: PreTrainedTokenizerFast, cluster_map_path: Optional[str] = None):
        self.args = args
        self.positive_size = args.positive_size
        self.negative_size = args.train_n_passages - self.positive_size
        assert self.negative_size > 0
        self.tokenizer = tokenizer

        # Load the cluster map if provided
        self.cluster_map = self._load_cluster_map(cluster_map_path) if cluster_map_path else None
        self.cluster_partitions = self._precompute_clusters() if self.cluster_map else None

        if self.args.data_name == 'esci':
            passage_file_name = 'passages.jsonl'
        elif self.args.data_name == 'msmarco':
            passage_file_name = 'passages.jsonl.gz'
        corpus_path = os.path.join(args.data_dir, passage_file_name)
        self.corpus: Dataset = load_dataset('json', data_files=corpus_path)['train']
        self.train_dataset, self.eval_dataset = self._get_transformed_datasets()

        # use its state to decide which positives/negatives to sample
        self.trainer: Optional[Trainer] = None

    def _load_cluster_map(self, cluster_map_path: str) -> Dict[int, int]:
        """Load the query_id to cluster_id mapping."""
        with open(cluster_map_path, 'r', encoding='utf-8') as f:
            return {int(k): v for k, v in json.load(f).items()}

    def _precompute_clusters(self) -> Dict[int, List[int]]:
        """Precompute partitions of query IDs by their cluster."""
        cluster_partitions = {}
        for query_id, cluster_id in self.cluster_map.items():
            if cluster_id not in cluster_partitions:
                cluster_partitions[cluster_id] = []
            cluster_partitions[cluster_id].append(query_id)
        return cluster_partitions

    def _get_cluster_shuffled_queries(self) -> List[int]:
        """Return queries shuffled within each cluster and ordered by clusters."""
        shuffled_queries = []
        for cluster_id, query_ids in self.cluster_partitions.items():
            random.shuffle(query_ids)  # Shuffle queries within the cluster
            shuffled_queries.extend(query_ids)  # Append the shuffled queries
        return shuffled_queries

    def _get_random_shuffled_queries(self, queries: List[int]) -> List[int]:
        """Return randomly shuffled queries globally."""
        shuffled_queries = queries[:]
        random.shuffle(shuffled_queries)
        return shuffled_queries

    def _transform_func(self, examples: Dict[str, List]) -> Dict[str, List]:
        current_epoch = int(self.trainer.state.epoch or 0)
        use_cluster_shuffle = self.cluster_map and current_epoch % 2 == 1  # Alternate epochs

        # Determine the query order for this epoch
        if use_cluster_shuffle:
            query_order = self._get_cluster_shuffled_queries()
        else:
            query_order = self._get_random_shuffled_queries(examples['query_id'])

        # Reorder examples based on the query order
        examples = {k: [v[query_order.index(qid)] for qid in query_order] for k, v in examples.items()}

        input_doc_ids: List[int] = group_doc_ids(
            examples=examples,
            positive_size=self.positive_size,
            negative_size=self.negative_size,
            offset=current_epoch + self.args.seed,
            available_doc_ids=list(range(len(self.corpus))),
            use_first_positive=self.args.use_first_positive,
            fill_neg_with_random_ids=self.args.fill_neg_with_random_ids
        )

        assert len(input_doc_ids) == len(examples['query']) * self.args.train_n_passages

        if self.args.data_name == 'esci':
            content = 'product_description'
            title = 'product_title'
        elif self.args.data_name == 'msmarco':
            content = 'contents'
            title = 'title'

        input_docs: List[str] = [self.corpus[doc_id][content] for doc_id in input_doc_ids]
        input_titles: List[str] = [self.corpus[doc_id][title] for doc_id in input_doc_ids]

        query_batch_dict = self.tokenizer(examples['query'],
                                          max_length=self.args.q_max_len,
                                          padding=PaddingStrategy.DO_NOT_PAD,
                                          truncation=True)
        doc_batch_dict = self.tokenizer(input_titles,
                                        text_pair=input_docs,
                                        max_length=self.args.p_max_len,
                                        padding=PaddingStrategy.DO_NOT_PAD,
                                        truncation=True)

        merged_dict = {'q_{}'.format(k): v for k, v in query_batch_dict.items()}
        step_size = self.args.train_n_passages
        for k, v in doc_batch_dict.items():
            k = 'd_{}'.format(k)
            merged_dict[k] = []
            for idx in range(0, len(v), step_size):
                merged_dict[k].append(v[idx:(idx + step_size)])

        if True:  # if self.args.do_kd_biencoder:
            qid_to_doc_id_to_score = {}

            def _update_qid_pid_score(q_id: str, ex: Dict):
                assert len(ex['doc_id']) == len(ex['score'])
                if q_id not in qid_to_doc_id_to_score:
                    qid_to_doc_id_to_score[q_id] = {}
                for doc_id, score in zip(ex['doc_id'], ex['score']):
                    qid_to_doc_id_to_score[q_id][int(doc_id)] = score

            for idx, query_id in enumerate(examples['query_id']):
                _update_qid_pid_score(query_id, examples['positives'][idx])
                _update_qid_pid_score(query_id, examples['negatives'][idx])

            merged_dict['kd_labels'] = []
            for idx in range(0, len(input_doc_ids), step_size):
                qid = examples['query_id'][idx // step_size]
                cur_kd_labels = [qid_to_doc_id_to_score[qid].get(doc_id, 0) for doc_id in input_doc_ids[idx:idx + step_size]]
                merged_dict['kd_labels'].append(cur_kd_labels)
            assert len(merged_dict['kd_labels']) == len(examples['query_id']), \
                '{} != {}'.format(len(merged_dict['kd_labels']), len(examples['query_id']))

        # Custom formatting function must return a dict
        return merged_dict

    def _get_transformed_datasets(self) -> Tuple:
        data_files = {}
        if self.args.train_file is not None:
            data_files["train"] = self.args.train_file.split(',')
        if self.args.validation_file is not None:
            data_files["validation"] = self.args.validation_file
        raw_datasets: DatasetDict = load_dataset('json', data_files=data_files)

        train_dataset, eval_dataset = None, None

        if self.args.do_train:
            if "train" not in raw_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = raw_datasets["train"]
            if self.args.max_train_samples is not None:
                train_dataset = train_dataset.select(range(self.args.max_train_samples))
            # Log a few random samples from the training set:
            for index in random.sample(range(len(train_dataset)), 3):
                logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
            train_dataset.set_transform(self._transform_func)

        if self.args.do_eval:
            if "validation" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = raw_datasets["validation"]
            eval_dataset.set_transform(self._transform_func)

        return train_dataset, eval_dataset
