import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique unique_words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing unique_words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        combined = positive + negative

        unique_words = set()
        for sentence in combined:
            for word in sentence.split():
                unique_words.add(word)
        unique_words = sorted(list(unique_words))
        
        word_id = {word: i + 1 for i, word in enumerate(unique_words)}

        encoded = [torch.tensor([word_id[word] for word in sentence.split()]) for sentence in combined]

        return nn.utils.rnn.pad_sequence(encoded, batch_first=True)
            

