from datasets import load_dataset
from ollama import Client

def main():
    client = Client(host="http://localhost:11434")
    model = "llama3.2:1b"

    # Load 20 examples from HotPotQA
    dataset = load_dataset("hotpot_qa", "distractor", split="train[:20]")

    for i, ex in enumerate(dataset, 1):
        q = ex["question"]
        ctx = "\n\n".join(sum(ex["context"]["sentences"], []))  # flatten context
        correct_answer = ex["answer"]

        # Generate the prompt and make the request to the model
        prompt = f"Context:\n{ctx}\n\nQuestion: {q}\n\nAnswer concisely:"
        resp = client.generate(model=model, prompt=prompt)

        # Print the formatted output 
        print(f"\nQ{i}: {q}")

        print(f"Generated Answer: {resp['response']}")
        print(f"Correct Answer: {correct_answer}\n")

if __name__ == "__main__":
    main()
