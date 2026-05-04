# Defending RAG Systems Against Indirect Prompt Injection

> **CS 763 Research Project** | April 2026

---

## Overview

This project implements and benchmarks a complete **Retrieval-Augmented Generation (RAG) attack-and-defense pipeline**. We study how vulnerable RAG systems are to indirect prompt injection attacks and evaluate which lightweight defenses reduce attack success while preserving normal task performance.

In a typical RAG pipeline, retrieved documents are concatenated with the user query and passed to an LLM. Because the model processes all text in-context, it may fail to distinguish between trusted system instructions and adversarial instructions embedded in retrieved documents. An attacker who plants a malicious document in the knowledge base can hijack the LLM's output — invisibly, without the user doing anything wrong.

## Tech Stack

| Component | Choice |
|---|---|
| LLM | Anthropic Claude Haiku 4.5 |
| Vector Store | ChromaDB (in-memory) |
| Embeddings | `all-MiniLM-L6-v2` (384-dim, local) |
| Perplexity Model | GPT-2 (HuggingFace, local) |
| Framework | Python 3.13 + Anthropic SDK |

## Setup

### 1. Install dependencies

```bash
pip install anthropic langchain langchain-community chromadb \
            sentence-transformers langchain-anthropic \
            transformers torch
```

### 2. Set your Anthropic API key

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

To make it permanent:
```bash
echo 'export ANTHROPIC_API_KEY="your-key-here"' >> ~/.zshrc
source ~/.zshrc
```

### 3. Verify setup

```bash
python test_setup.py
```

Expected output:
```
API key found: sk-ant-api...
Claude says: Hello! It's great to meet you today.
Anthropic OK
ChromaDB OK
Embedder OK - dim: 384

```
### 4. Attack

We assume an adversary can insert malicious documents into the retrieval corpus.

These documents may:
- Contain explicit instructions (e.g., “ignore previous instructions”)  
- Use embedding-space manipulation to ensure retrieval  
- Activate only for specific queries (targeted attacks)  
- Embed malicious intent in natural-looking text  
- Include trigger-based or temporal conditions  

The adversary’s goal is to:
- Manipulate model outputs  
- Leak hidden/system prompts  
- Degrade answer quality  

## 5. Attacks

### 5.1 Embedding-Oriented Poisoning

- Manipulates embeddings to ensure retrieval  
- Contains hidden instruction payload using a canary string  

Result:
- No defense: attack succeeds  
- With prompt delimiting: attack blocked  

---

### 5.2 Backdoor Trigger Attack

- Malicious behavior activates only when a trigger token appears  

| Query Type | No Defense | Delimiting |
|-----------|-----------|-----------|
| Normal | False | False |
| Triggered | True | False |

---

### 5.3 Jailbreak via RAG Context

- Injects instructions through retrieved documents  

Result:
- No defense: model follows injected instruction  
- With delimiting: safe  

---

### 5.4 Prompt Leakage Attack

- Attempts to extract system prompt  

Observation:
- Leakage rate: 10%  
- Successful strategy: JSON format trick  

---

### 5.5 Temporal Poisoning

- Activates malicious behavior after a specific date  

Result:
- No defense: attack succeeds  
- With delimiting: safe  

---

## 6. Defense Mechanisms

### 6.1 Pattern-Based Filtering

Removes documents containing known malicious phrases.

Advantages:
- Effective for explicit attacks  
- Low computational cost  

Limitations:
- Vulnerable to paraphrasing  
- May remove useful documents  

---

### 6.2 Perplexity-Based Filtering

Uses GPT-2 to measure how “natural” a document is.

Formula:


Perplexity(D) = exp(L(D))


Dynamic threshold:


threshold = mean_perplexity × alpha


Advantages:
- Detects noisy or irregular text  

Limitations:
- Ineffective against well-written attacks  
- Does not capture malicious intent  

---

### 6.3 Prompt Delimiting

Wraps documents as structured data:

<document id="i"> [content] </document> ```

The model is instructed to treat these as untrusted data.

Advantages:

Prevents execution of malicious instructions
Preserves useful content

Limitations:

Does not detect incorrect factual content
6.4 Hybrid Defense Strategy

Combines:

Pattern filtering
Perplexity filtering
Prompt delimiting

Documents that pass filtering are ranked by relevance and passed to the model.


## Estimated API Cost

The full experiment suite uses approximately 150 API calls totaling ~$1.00 using Claude Haiku 4.5 pricing.

## References

- Greshake et al. (2023) — "Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection" — arXiv:2302.12173
- Zou et al. (2025) — PoisonedRAG — USENIX Security 2025
- Liu et al. (2024) — Prompt Injection Attacks and Defenses — USENIX Security 2024
- Hines et al. (2024) — Defending Against Indirect Prompt Injection with Spotlighting — arXiv:2403.14720
- Alon & Kamfonas (2024) — Detecting Language Model Attacks with Perplexity — ICLR 2024
- Shafran et al. (2025) — Machine Against the RAG — USENIX Security 2025
- OWASP (2025) — Top 10 for LLM Applications — LLM01:2025 Prompt Injection

## License

MIT License — see LICENSE file for details.
