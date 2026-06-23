import networkx as nx

class SkillGraph:
    def __init__(self):
        self.graph = nx.Graph()
        
        self.categories_map = {
            "Vector Database": [
                "Pinecone", "Milvus", "Weaviate", "FAISS", 
                "ChromaDB", "Qdrant", "HNSW", "Annoy"
            ],
            "Embeddings-based retrieval": [
                "Pinecone", "Milvus", "Weaviate", "FAISS", 
                "ChromaDB", "Qdrant", "HNSW", "Annoy"
            ],
            "Deep Learning": ["PyTorch", "TensorFlow", "Keras", "JAX"],
            "Neural Networks": ["PyTorch", "TensorFlow", "Keras", "JAX"],
            "Machine Learning": ["Scikit-learn", "XGBoost", "LightGBM", "Pandas", "NumPy"],
            "LLM Orchestration": ["LangChain", "LlamaIndex", "Haystack"],
            "Generative AI": ["GPT-4", "Claude", "Gemini", "Llama 3", "Mistral"],
            "Cloud Infrastructure": ["AWS", "Azure", "GCP", "Kubernetes", "Docker", "Terraform"],
        }
        
        # Populate the bipartite graph with edges from specific tools to their category names
        for cat, tools in self.categories_map.items():
            for tool in tools:
                self.graph.add_edge(tool, cat)
        
        # Create lowercase mappings for case-insensitive lookup
        self.nodes_lower = {str(n).lower(): n for n in self.graph.nodes()}
        self.categories_lower = {cat.lower(): cat for cat in self.categories_map}
        
        self.tools_lower = {}
        for tools in self.categories_map.values():
            for t in tools:
                self.tools_lower[t.lower()] = t

    def expand_skill(self, skill: str) -> list[str]:
        """
        Given a skill, find related skills/equivalents.
        
        If it's a category (e.g. "Vector Database"), returns the tools in that category
        and any synonym category names.
        If it's a tool (e.g. "Pinecone"), returns its categories and equivalent tools in those categories.
        """
        skill_lower = skill.strip().lower()
        
        # Exact node match
        actual_node = None
        if skill_lower in self.nodes_lower:
            actual_node = self.nodes_lower[skill_lower]
        else:
            # Fallback substring matching
            for nl, an in self.nodes_lower.items():
                if skill_lower in nl or nl in skill_lower:
                    actual_node = an
                    skill_lower = nl
                    break
        
        if not actual_node:
            return []
            
        expanded = set()
        
        # Determine if it's a category node or a tool node
        if skill_lower in self.categories_lower:
            # Get tools in this category
            tools = list(self.graph.neighbors(actual_node))
            expanded.update(tools)
            # Add sibling categories that share these tools
            for tool in tools:
                for cat in self.graph.neighbors(tool):
                    expanded.add(cat)
        elif skill_lower in self.tools_lower:
            # Get categories this tool belongs to
            cats = list(self.graph.neighbors(actual_node))
            expanded.update(cats)
            # Add all other tools in those categories
            for cat in cats:
                expanded.update(self.graph.neighbors(cat))
        else:
            # Fallback networkx neighbors lookup
            expanded.update(self.graph.neighbors(actual_node))
            
        if actual_node in expanded:
            expanded.remove(actual_node)
            
        return sorted(list(expanded))

    def expand_skills(self, skills: list[str]) -> dict[str, list[str]]:
        """
        Expand a list of skills, returning a mapping of original skill to related skills.
        """
        expansion = {}
        for skill in skills:
            expanded = self.expand_skill(skill)
            if expanded:
                expansion[skill] = expanded
        return expansion
