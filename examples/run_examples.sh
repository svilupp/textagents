#!/bin/bash
# Example CLI commands for TextAgents

set -e

echo "=== Running minimal_judge with inline input ==="
uv run textagents run examples/minimal_judge.txt --input "The sky is blue."

echo ""
echo "=== Running safety_judge with inline inputs ==="
uv run textagents run examples/safety_judge.txt \
  --user_input "What is 2+2?" \
  --model_output "2+2 equals 4."

echo ""
echo "=== Running safety_judge with file inputs ==="
uv run textagents run examples/safety_judge.txt \
  --user_input @examples/inputs/sample_user_input.txt \
  --model_output @examples/inputs/sample_model_output.txt

echo ""
echo "=== Running sentiment_analyzer ==="
uv run textagents run examples/sentiment_analyzer.txt \
  --text "I absolutely loved this product! Best purchase ever."

echo ""
echo "=== Show agent info ==="
uv run textagents info examples/safety_judge.txt

echo ""
echo "=== Validate agent definition ==="
uv run textagents validate examples/minimal_judge.txt
