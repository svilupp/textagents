# Sentiment Analyzer

Classify text sentiment with confidence scores.

```toml title="sentiment.txt"
---
[agent]
model = "openai:gpt-4o"

[agent.settings]
temperature = 0

[agent.output_type.reasoning]
type = "str"
max_length = 300
description = "Explanation."

[agent.output_type.sentiment]
type = "str"
enum = ["positive", "negative", "neutral", "mixed"]

[agent.output_type.confidence]
type = "float"
ge = 0.0
le = 1.0

[agent.output_type.key_phrases]
type = "list[str]"
optional = true
max_items = 5
---
Analyze the sentiment:

{text}
```

## Usage

```python
import textagents

agent = textagents.load_agent("sentiment.txt")
result = agent.run_sync(text="I love this product!")

print(result.sentiment)    # "positive"
print(result.confidence)   # 0.95
print(result.key_phrases)  # ["love", "this product"]
```

## Batch Analysis

```python
import asyncio
import textagents

async def analyze_batch(texts):
    agent = textagents.load_agent("sentiment.txt")
    for text in texts:
        result = await agent.run(text=text)
        print(f"{result.sentiment:8} ({result.confidence:.0%}): {text[:40]}...")

asyncio.run(analyze_batch([
    "Great product!",
    "Terrible experience.",
    "It's okay.",
]))
```
