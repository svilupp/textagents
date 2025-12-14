# CLI

Run agents from the command line.

## Commands

### run

```bash
uv run textagents run <agent_file> [--<input> VALUE] [--inputs FILE] [--format json|pretty] [--model MODEL]
```

```bash
# Inline inputs
uv run textagents run judge.txt --statement "The sky is blue."

# From JSON file
uv run textagents run judge.txt --inputs data.json

# Load input from file
uv run textagents run judge.txt --text @document.txt

# Override model
uv run textagents run judge.txt --text "test" --model openai:gpt-4o-mini
```

### info

```bash
uv run textagents info judge.txt
```

Shows agent configuration, inputs, and output fields.

### validate

```bash
uv run textagents validate judge.txt
```

Validates agent definition without running. Exit code 0 if valid.

## Environment Variables

The CLI loads `.env` automatically:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```
