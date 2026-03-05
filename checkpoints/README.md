# My Fine‑Tuned Model

**Overview**

This model was fine‑tuned with Reinforcement Learning from Human Feedback using TRL.

**Usage**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("your-username/your-model-name")
tokenizer = AutoTokenizer.from_pretrained("your-username/your-model-name")
