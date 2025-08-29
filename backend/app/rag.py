import os, glob, re
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def read_texts(base_path: str) -> Dict[str, str]:
    docs = {}
    for path in glob.glob(os.path.join(base_path, "*.md")):
        with open(path, "r", encoding="utf-8") as f:
            docs[os.path.basename(path)] = f.read()
    return docs

class KB:
    def __init__(self, base_path: str = "kb"):
        self.base_path = base_path
        self.docs = read_texts(base_path)
        self.names = list(self.docs.keys())
        self.corpus = [self.docs[n] for n in self.names]
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        if len(self.corpus) == 0:
            self.X = None
        else:
            self.X = self.vectorizer.fit_transform(self.corpus)

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        if self.X is None:
            return []
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.X)[0]
        idxs = sims.argsort()[::-1][:top_k]
        results = []
        for i in idxs:
            name = self.names[i]
            text = self.docs[name]
            # produce a short snippet (first 50 words)
            words = re.findall(r"\S+", text)
            snippet = " ".join(words[:60]) + ("..." if len(words) > 60 else "")
            results.append({
                "doc": name,
                "score": float(sims[i]),
                "snippet": snippet
            })
        return results
