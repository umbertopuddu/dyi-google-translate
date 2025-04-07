# 🌍 Google-like Translate Transformer — *Proof of Concept*

> A Transformer-based neural machine translation system for Italian → English.  
> Trained on Europarl v7 (Parliament Proceedings) with PyTorch, SentencePiece, and a custom-built transformer architecture.  
> Fine-tuned overnight on **NVIDIA H100 SXM** for **$20 on [Vast.ai](https://vast.ai/).

---

## 📁 Project Structure

```
.
├── README.md
├── requirements.txt
├── saved_model/            # Stores the best checkpoint: best_ckpt.tar
├── sp/                     # SentencePiece models and vocabs
│   ├── src_sp.model
│   ├── src_sp.vocab
│   ├── trg_sp.model
│   └── trg_sp.vocab
├── src/
│   ├── run.py              # Main entrypoint for inference
│   ├── constants.py        # Paths, vocab size, special tokens
│   ├── custom_data.py      # Tokenizer + DataLoader logic
│   ├── data_structure.py   # BeamNode definition
│   ├── transformer.py      # Full Transformer model
│   ├── layers.py           # Attention, FFN layers
│   └── sentencepiece_train.py  # (Optional) for generating SP models
```

---

## ⚙️ Features

- ✅ Transformer (Encoder-Decoder) with Positional Encoding
- ✅ SentencePiece tokenization
- ✅ Beam Search & Greedy Decoding
- ✅ Multi-GPU support (`nn.DataParallel`)
- ✅ Training/Inference logging with timestamps
- ✅ Modular, readable codebase
- ✅ Works well on formal Italian input (legal/gov data)

---

## 📚 Dataset & Training

- 🔗 Dataset: [European Parliament Proceedings Parallel Corpus](https://www.statmt.org/europarl/)
- 🗃 Size: ~2.1 million aligned sentence pairs
- 🖥️ Hardware: **NVIDIA H100 SXM (Vast.ai)**
- ⏱️ Duration: ~10 epochs overnight
- 💰 Budget: ~\$20

> Because the dataset is formal, the model performs best when translating well-structured, formal Italian.

---

## 🧪 Inference Usage

### 🔧 Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

### ▶️ Run Translation

From the project root:

```bash
python src/run.py --model 'best_ckpt.tar' --input 'come stai?' --decode 'beam'
```

Or use greedy decoding:

```bash
python src/run.py --model 'best_ckpt.tar' --input 'come stai?' --decode 'greedy'
```

### 🧾 Output

```
Input: come stai?
Result: How are you?'??' How are you?
Inference finished! || Total inference time: 0mins 3secs
```

> 🚨 Note: Output may include repetitions or struggle with proper punctuation — common in smaller Transformer models. That said, **semantic understanding is often surprisingly accurate.**

---

## 🔄 Example: Beam vs Greedy (Real Outputs)

| Input                          | Greedy Output (Truncated)                                    | Beam Output (Truncated)                             |
|-------------------------------|---------------------------------------------------------------|------------------------------------------------------|
| `come stai?`                  | How are you?'??' How are you???????????????...               | How are you?'??' How are you?                        |
| `sto cercando lavoro`         | I am looking for work job job job job searching job...        | I am looking for work job job job job searching...   |
| `mi chiamo Umberto`           | I am called Umberttobertoberto ometta ometta ometta...        | I am called Umberto Umbertoto Umbertoto be...        |
| `come sta Signor Rossi?`      | How is Mr Rossi?? what is Mr Rosss????? what is Mr...         | How is Mr Rossi?? what is Mr Rossite?                |

---

## 🛠️ How It Works

### During Training

- Loads SentencePiece tokenizers (`sp/src_sp.model`, `sp/trg_sp.model`)
- Loads vocab files (`.vocab`)
- Initializes a Transformer model with:
  - Self-attention encoder
  - Cross-attention decoder
  - Custom padding/masking logic
- Optimizes with `NLLLoss` and `Adam`
- Saves best checkpoint to `saved_model/best_ckpt.tar`

### During Inference

- Tokenizes input with SentencePiece
- Passes it through encoder
- Decodes using greedy or beam search
- Detokenizes the final sequence

---

## 🚀 Future Improvements

- Train on full 300M-line dataset
- Fine-tune with general-domain text (e.g. Wikipedia)
- Add punctuation/stop-token rewards (to prevent hallucinations)
- Compress model for faster inference
- Build web UI with `Streamlit`

---

## 🙌 Contributors Welcome

> 🤝 **Looking for contributors**  
> If you're into building affordable AI, experimenting with language models, or just enjoy hacking with transformers — you're welcome to help extend this. More languages, more robustness, more research — let’s do it together.

---

## 📄 License & Credits

- 🗂 **Corpus**: [Europarl v7 (StatMT.org)](https://www.statmt.org/europarl/)
- 💻 Framework: PyTorch
- 📚 Tokenization: SentencePiece
- 🧠 Author: [Umberto Puddu](https://github.com/umbertopuddu)
- ⚖️ License: MIT (default)