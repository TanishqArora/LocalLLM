# =============================================
# README — how to build & run
# =============================================
# 1) Make sure NVIDIA Container Toolkit is installed on the host
#    https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
#    Quick check: `docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi`
#
# 2) Ensure the model is downloaded on the host at ./qwen2.5-7b-instruct
#    huggingface-cli login
#    huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./qwen2.5-7b-instruct --resume-download
#
# 3) Build the image
#    docker compose build
#
# 4) Run the server (GPU pass-through)
#    docker compose up -d
#    # Logs: docker compose logs -f qwen-server
#
# 5) Test the health endpoint
#    curl http://localhost:8000/health
#
# 6) Generate text
#    curl -s http://localhost:8000/generate \
#      -H "Content-Type: application/json" \
#      -d '{
#        "agent_id": "analyst",
#        "system": "You are concise and factual.",
#        "context": "Qwen2.5-7B supports long context and strong reasoning.",
#        "messages": [{"role":"user","content":"Summarize the context in one sentence."}],
#        "max_new_tokens": 128,
#        "temperature": 0.2
#      }' | jq .
#
# 7) Stop & remove
#    docker compose down
#
# Notes:
# - Change MODEL_PATH/LOAD_4BIT via environment vars in docker-compose.yml.
# - To switch models, mount a different host folder to /models and set MODEL_PATH accordingly.
# - If you want 8-bit instead of 4-bit, set LOAD_4BIT=0.
# - If you see CUDA OOM, keep 4-bit on and reduce max_new_tokens.