from typing import List
from openai import OpenAI, OpenAIError

from .models import Source


class LLMClient:
    def __init__(self, api_key: str | None):
        self.enabled = api_key is not None
        self.client = OpenAI(api_key=api_key) if api_key else None

    def answer(self, question: str, sources: List[Source]) -> str:
        if not self.enabled or not self.client:
            return self._fallback_answer(question, sources)
        context = "\n\n".join([f"[{s.id}] {s.text}" for s in sources])
        system_prompt = (
            "You are a support assistant. Answer using the provided context. "
            "If the answer is not in context, say you do not know."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=256,
                temperature=0.7,
            )
            return completion.choices[0].message.content or self._fallback_answer(question, sources)
        except (OpenAIError, Exception):
            return self._fallback_answer(question, sources)

    @staticmethod
    def _fallback_answer(question: str, sources: List[Source]) -> str:
        if not sources:
            return "No context available to answer this question."
        joined = " ".join(s.text for s in sources)[:500]
        return f"(Fallback) Based on context: {joined}"
