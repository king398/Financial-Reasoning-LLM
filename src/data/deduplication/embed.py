import torch.nn.functional as F
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from typing import Tuple, List
from  tqdm import tqdm
def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    """
    Computes the average pooled embedding across the sequence dimension,
    taking into account the attention mask to ignore padded tokens.

    This function performs mean pooling over the token embeddings in
    `last_hidden_states`, masking out positions corresponding to padding
    (where the attention mask is 0). It sums only the valid (non-masked)
    embeddings and divides by the number of valid tokens per sequence.

    Args:
        last_hidden_states (Tensor):
            A tensor of shape (batch_size, seq_len, hidden_dim) containing
            the hidden states output from a transformer model.
        attention_mask (Tensor):
            A tensor of shape (batch_size, seq_len) where each element is 1
            for valid tokens and 0 for padding tokens.

    Returns:
        Tensor:
            A tensor of shape (batch_size, hidden_dim) representing the
            average pooled embeddings for each input sequence.
    """
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def load_model(model_name: str = "intfloat/e5-small-v2") -> Tuple[AutoTokenizer, AutoModel]:
    """
    Loads a pretrained transformer model and its tokenizer from the Hugging Face Hub.

    The function automatically detects whether a CUDA-enabled GPU is available
    and places the model on the appropriate device (GPU or CPU). The model is
    loaded in half-precision (float16) for improved memory efficiency when
    running on GPU.

    Args:
        model_name (str, optional):
            The Hugging Face model identifier or local path of the pretrained model.
            Defaults to `"intfloat/e5-small-v2"`.

    Returns:
        tokenizer (AutoTokenizer):
            The tokenizer corresponding to the specified model.
        model (AutoModel):
            The loaded transformer model ready for inference or fine-tuning.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, dtype=torch.float16, device_map=device)
    return tokenizer, model

def prepare_documents(documents: List[str], prefix:str="query:") -> List[str]:
    return [f"{prefix} {x}" for x in documents]

def embed_documents(documents: List[str],
                    model: AutoModel,
                    tokenizer: AutoTokenizer,
                    batch_size: int = 16,
                    prefix: str = "query:") -> Tensor:
    embeddings = []
    for i in tqdm(range(0, len(documents), batch_size)):
        batch = prepare_documents(documents[i:i + batch_size],prefix)
        inputs = tokenizer(batch, max_length=512, padding=True, truncation=True, return_tensors='pt')
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(**inputs)
        batch_embeddings = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
        batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
        embeddings.append(batch_embeddings.detach().cpu().float())
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings
