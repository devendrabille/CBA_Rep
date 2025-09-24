#!/usr/bin/env python3
# Local conversational chatbot using a small HF model (no API keys, offline)
# Model suggestions: "microsoft/DialoGPT-small" (compact), or "facebook/blenderbot_small-90M"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "microsoft/DialoGPT-small"  # swap to another local model if you like

def load_model(name=MODEL_NAME):
    print(f"Loading model: {name} (first run will download; later runs are offline)")
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    model.eval()
    return tokenizer, model

def generate_reply(tokenizer, model, chat_history_ids, user_text, max_new_tokens=120):
    # Prepare input
    new_user_input_ids = tokenizer.encode(user_text + tokenizer.eos_token, return_tensors="pt")
    if chat_history_ids is None:
        bot_input_ids = new_user_input_ids
    else:
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)

    # Generate
    output_ids = model.generate(
        bot_input_ids,
        max_length=min(1024, bot_input_ids.shape[-1] + max_new_tokens),
        do_sample=True,
        top_p=0.92,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    # Extract only the newly generated tokens
    new_tokens = output_ids[:, bot_input_ids.shape[-1]:]
    reply = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
    return reply, output_ids

def main():
    tokenizer, model = load_model()
    print("Local LLM Chatbot (DialoGPT-small). Type 'exit' to quit.")
    chat_history_ids = None

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye! 👋")
            break

        if not user:
            continue
        if user.lower() in {"exit", "quit", "bye"}:
            print("Bot: Goodbye! 👋")
            break

        reply, chat_history_ids = generate_reply(tokenizer, model, chat_history_ids, user)
        print(f"Bot: {reply}")

if __name__ == "__main__":
    main()
