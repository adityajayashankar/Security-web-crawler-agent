"""
main.py
-------
Entry point for the vulnerability analysis agent.
Run: python main.py
"""
from dotenv import load_dotenv
load_dotenv()
from pipeline.langgraph_agent import run_agent


def main():
    print("=" * 60)
    print("  Vulnerability Analysis Agent")
    print("  Powered by fine-tuned Mistral-7B (6-layer vuln dataset)")
    print("=" * 60)
    print("Type a CVE ID or a vulnerability question.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("Goodbye.")
            break
        if not user_input:
            continue

        print("\n🤖 Agent working...\n")
        result = run_agent(user_input, verbose=True)

        print("\n" + "=" * 60)
        print("📋 FINAL REPORT")
        print("=" * 60)
        print(result)
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
