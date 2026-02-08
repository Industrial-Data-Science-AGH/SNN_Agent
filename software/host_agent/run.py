def infer(prompt: str) -> str:
    return f"[mock] {prompt}"
if __name__ == "__main__":
    print(infer("Hello IDS"))
