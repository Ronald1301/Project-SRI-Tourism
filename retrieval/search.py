class SemanticSearcher:
    def __init__(self):
        # Cargar índice TF-IDF
        self.tfidf_index = TFIDFIndex.load(...)
        
        # Cargar modelo LSI
        self.lsi_model = LSIModel.load()
        
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Busca documentos semánticamente similares a la query.
        
        Pasos:
        1. Preprocesar query → tokens
        2. Vectorizar con TF-IDF → query_vector
        3. Proyectar a espacio LSI → query_lsi
        4. Calcular similitud coseno con doc_vectors
        5. Retornar top-k
        """
        # 1. Tokenizar + preprocesar
        tokens = preprocess(query)
        
        # 2. Vectorizar con TF-IDF
        query_vector = self.tfidf_index.vectorize_query(tokens)
        
        # 3. Proyectar a LSI
        query_lsi = self.lsi_model.transform_query(query_vector)
        
        # 4. Similitud coseno
        similarities = cosine_similarity(query_lsi, self.lsi_model.doc_vectors)
        
        # 5. Ranking
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(self.doc_ids[i], similarities[i]) for i in top_indices]