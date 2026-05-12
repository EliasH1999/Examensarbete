import os
print("OPENAI_API_KEY set:", bool(os.getenv("OPENAI_API_KEY")))
print("First chars:", os.getenv("OPENAI_API_KEY","")[:5])