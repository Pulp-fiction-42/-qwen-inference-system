cd /home/bgi-chao/inference
make -j4 repl

weights=/home/bgi-chao/wdr/main/model.bin
vocab=/home/bgi-chao/llama-deploy/models/qwen2.5-1.5b-instruct-q8_0.gguf

printf 'hello, world\nwhat is CUDA?\nsummarize your last answer in one sentence\n/exit\n' |
  ./build/qwen_repl "$weights" "$vocab" \
    --max-new-tokens 320 --context-limit 2048 >out.txt 2>metrics_raw.txt

# Each turn starts with its "  Tokenizer Encode:" line; insert a header before it.
awk '/^  Tokenizer Encode:/ { n++; print "--- Turn " n " ---" } { print }' metrics_raw.txt > metrics.txt

cat metrics.txt