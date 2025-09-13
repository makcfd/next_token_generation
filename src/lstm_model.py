
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
from pathlib import Path
from transformers import AutoTokenizer


class LSTMLM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers=1, pad_id=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, lengths):
        x = self.emb(input_ids)  # B×L×E
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        out_padded, _ = pad_packed_sequence(packed_out, batch_first=True)
        logits = self.head(out_padded)  # B×L×V
        return logits

    @torch.no_grad()
    def generate(self, prefix_ids: torch.Tensor, max_new_tokens: int, eos_id: int):
        self.eval()
        B, L0 = prefix_ids.shape
        lengths = (prefix_ids != self.emb.padding_idx).sum(dim=1)
        x = self.emb(prefix_ids)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h, c) = self.rnn(packed)  # get lst hidden state for each sequnce
        seq = prefix_ids.clone()
        for _ in range(max_new_tokens):
            # last non-pad token for each sequence:
            last_tokens = []
            for i in range(B):
                li = lengths[i].item()
                last_tokens.append(seq[i, li-1:li])
            last_tokens = torch.vstack(last_tokens)  # B×1
            x1 = self.emb(last_tokens)              # B×1×E
            out, (h, c) = self.rnn(x1, (h, c))      # B×1×H
            logits = self.head(out)                 # B×1×V
            next_id = logits.argmax(dim=-1)         # B×1
            # append
            seq = torch.cat([seq, next_id], dim=1)
            lengths = lengths + (next_id != self.emb.padding_idx).squeeze(1).long()
            # early stop if hit eos token
            if (next_id.squeeze(1) == eos_id).all():
                break
        return seq


def load_lstm_lm(run_dir: str, device=None):
    run_dir = Path(run_dir)
    device = device

    tokenizer = AutoTokenizer.from_pretrained(run_dir)

    ckpt = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    cfg  = ckpt.get("config", {})
    state = ckpt["model_state"]

    model = LSTMLM(
        vocab_size=tokenizer.vocab_size,
        emb_dim=cfg.get("emb_dim", 256),
        hidden_dim=cfg.get("hidden_dim", 512),
        num_layers=cfg.get("num_layers", 1),
        pad_id=cfg.get("pad_id", tokenizer.pad_token_id)
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    pad_id = cfg.get("pad_id", tokenizer.pad_token_id)
    eos_id = cfg.get("eos_id", tokenizer.sep_token_id)

    return tokenizer, model, pad_id, eos_id