from pathlib import Path
import subprocess

from transformers import AutoTokenizer


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path("/home/bgi-chao/wdr/main/Qwen2.5-1.5B-Instruct")
WEIGHTS = Path("/home/bgi-chao/wdr/main/model.bin")
MAX_SEQ_LEN = 2048  # Must match run.cu.


def main():
    # Startup: load the tokenizer and launch one persistent inference process.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    messages = []  # Conversation history for this REPL session.

    engine = subprocess.Popen(
        [str(PROJECT_DIR / "engine"), str(WEIGHTS)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )

    try:
        while True:
            # 1. Read the user's text.
            prompt = input("\nYou: ")
            if prompt.strip() == "/quit":
                break
            if not prompt.strip():
                continue

            # 2. Tokenize the full conversation, including the new user message.
            messages.append({"role": "user", "content": prompt})
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            )
            token_ids = encoded["input_ids"]
            if len(token_ids) > MAX_SEQ_LEN:
                messages.pop()  # The rejected input does not enter the history.
                print("Conversation exceeds 2048 tokens. Shorten the input or restart.")
                continue

            # 3. Send one complete request.
            ids_text = " ".join(str(token) for token in token_ids)
            engine.stdin.write(f"{len(token_ids)} {ids_text}\n")
            engine.stdin.flush()

            # 4. Receive generated IDs and print newly decoded text.
            print("Assistant: ", end="", flush=True)
            generated_ids = []
            text = ""  # Also handles DONE without any generated tokens.
            printed_length = 0

            while True:
                line = engine.stdout.readline()
                if not line:
                    raise RuntimeError("Inference process exited unexpectedly.")
                if line.strip() == "DONE":
                    break

                generated_ids.append(int(line))
                text = tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                # A token may contain only part of a UTF-8 character.
                if text.endswith("\ufffd"):
                    continue

                print(text[printed_length:], end="", flush=True)
                printed_length = len(text)

            # Flush any remaining text at the end of the response.
            if generated_ids:
                print(text[printed_length:], end="", flush=True)
            print()
            messages.append({"role": "assistant", "content": text})

    finally:
        engine.stdin.close()
        if engine.poll() is None:
            engine.terminate()
        engine.wait()


if __name__ == "__main__":
    main()
