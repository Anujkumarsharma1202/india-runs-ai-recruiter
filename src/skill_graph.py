import networkx as nx

class SkillGraph:
    def __init__(self):
        self.graph = nx.Graph()
        
        # Add edges (Skill, Category)
        edges = [
            ("PyTorch", "Deep Learning"),
            ("TensorFlow", "Deep Learning"),
            ("Keras", "Deep Learning"),
            ("Pinecone", "Vector DB"),
            ("Milvus", "Vector DB"),
            ("Weaviate", "Vector DB"),
            ("FAISS", "Vector DB"),
            ("React", "Frontend"),
            ("Vue", "Frontend"),
            ("Angular", "Frontend"),
            ("AWS", "Cloud"),
            ("Azure", "Cloud"),
            ("GCP", "Cloud"),
        ]
        self.graph.add_edges_from(edges)
        
        # Create a lowercase mapping for case-insensitive lookup
        self.nodes_lower = {str(n).lower(): n for n in self.graph.nodes()}

    def expand_skill(self, skill: str) -> list[str]:
        """
        Given a skill, find related skills (siblings in the graph under the same category).
        """
        skill_lower = skill.strip().lower()
        
        # If the skill itself matches a node exactly (case-insensitive)
        if skill_lower in self.nodes_lower:
            actual_node = self.nodes_lower[skill_lower]
            
            # Find neighbors (which are categories)
            categories = list(self.graph.neighbors(actual_node))
            
            if not categories:
                # If the node has no neighbors, or is a category itself?
                # Actually, our graph is bipartite. Let's assume skill -> category -> skill
                # If it's a category, neighbors are skills. Let's just return all its neighbors.
                return list(self.graph.neighbors(actual_node))
            
            expanded = set()
            for cat in categories:
                # Get other skills in the same category
                expanded.update(self.graph.neighbors(cat))
                
            # Remove the original skill itself
            if actual_node in expanded:
                expanded.remove(actual_node)
                
            return list(expanded)
            
        # Fallback: check if the input skill name is a substring of any node
        for node_lower, actual_node in self.nodes_lower.items():
            if skill_lower in node_lower or node_lower in skill_lower:
                categories = list(self.graph.neighbors(actual_node))
                expanded = set()
                if not categories:
                    expanded.update(self.graph.neighbors(actual_node))
                else:
                    for cat in categories:
                        expanded.update(self.graph.neighbors(cat))
                if actual_node in expanded:
                    expanded.remove(actual_node)
                if expanded:
                    return list(expanded)
                    
        return []

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
