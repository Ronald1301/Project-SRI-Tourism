"""
Latent Semantic Indexing (LSI) model for the Information Retrieval system.

This module implements LSI using Truncated SVD to project documents
from a high-dimensional TF-IDF space into a lower-dimensional latent
semantic space. Core concepts:

- SVD decomposition: U Σ V^T where Σ ≈ diagonal matrix of singular values
- Dimensionality reduction: retain only k largest singular values
- Document vectors: rows of U_k × Σ_k (semantic representation)
- Query projection: q_semantic ← (q_tfidf × V_k^T)

Benefits:
- Reduces noise and sparsity of TF-IDF
- Captures latent semantic structure
- Improves retrieval by semantic similarity (not just term overlap)

This module is part of the retrieval layer of the SRI project.
It integrates with the TF-IDF indexer and is used by the search module.
"""

import json
import os
import pickle

import numpy as np
from sklearn.decomposition import TruncatedSVD

from utils.file_manager import load_json, load_numpy, load_pickle, save_json, save_numpy, save_pickle


class LSIModel:
    """
    Latent Semantic Indexing model based on Truncated SVD.
    
    This class transforms high-dimensional TF-IDF vectors into a lower-dimensional
    latent semantic space. It maintains:
    
    Attributes:
        n_components (int): dimension of the latent semantic space
        svd_model (TruncatedSVD): trained SVD decomposer
        doc_vectors (ndarray): semantic representations of documents (n_docs × n_components)
        is_trained (bool): whether the model has been successfully trained
        
    The model follows the classical LSI pipeline and is designed to be
    compatible with cosine similarity-based retrieval.
    """
    
    def __init__(self, n_components: int = 100):
        """
        Initialize the LSI model.
        
        Args:
            n_components (int): Number of latent semantic dimensions (topics).
                              Default: 100.
                              Recommended: 50-300 depending on corpus size.
                              
        Raises:
            ValueError: if n_components is not a positive integer.
        """
        if not isinstance(n_components, int) or n_components <= 0:
            raise ValueError("n_components must be a positive integer")
        
        self.n_components = n_components
        self.svd_model = None
        self.doc_vectors = None
        self.is_trained = False
        
    def train(self, tfidf_matrix: np.ndarray) -> "LSIModel":
        """
        Train the LSI model on a TF-IDF matrix.
        
        Process:
        1. Validates input TF-IDF matrix
        2. Fits TruncatedSVD on the matrix
        3. Computes semantic document vectors via SVD transformation
        4. Stores the learned representations
        
        Args:
            tfidf_matrix (ndarray): TF-IDF document-term matrix
                                   Shape: (num_documents, num_terms)
                                   Type: float (typically from sklearn's TfidfVectorizer)
                                   
        Returns:
            LSIModel: self for method chaining
            
        Raises:
            ValueError: if matrix is empty, has invalid shape, or has dtype issues
            ValueError: if n_components > min(matrix dimensions)
            RuntimeError: if SVD fitting fails
            
        Notes:
            - The actual semantic space has dimension min(n_components, min(m, n))
              where m = num_documents and n = num_terms.
            - SVD uses randomized solver for efficiency on large matrices.
        """
        self._validate_tfidf_matrix(tfidf_matrix)
        self._validate_components_vs_matrix(tfidf_matrix)
        
        try:
            # Initialize Truncated SVD with randomized solver for efficiency
            self.svd_model = TruncatedSVD(
                n_components=self.n_components,
                n_iter=100,
                random_state=42
            )
            
            # Fit SVD on TF-IDF matrix
            # Returns: U × Σ (document vectors in latent space)
            self.doc_vectors = self.svd_model.fit_transform(tfidf_matrix)
            
            self.is_trained = True
            return self
            
        except Exception as e:
            raise RuntimeError(f"SVD training failed: {str(e)}") from e
    
    def transform_query(self, query_vector: np.ndarray) -> np.ndarray:
        """
        Project a query vector to the LSI semantic space.
        
        Mathematical operation:
        q_lsi = q_tfidf × V^T
        where V^T are the components learned by TruncatedSVD.
        
        This allows comparison between the query and document semantic vectors
        using standard similarity measures (e.g., cosine similarity).
        
        Args:
            query_vector (ndarray): query TF-IDF vector
                                   Shape: (num_terms,) or (1, num_terms)
                                   Must have same dimension as training matrix columns
                                   
        Returns:
            ndarray: query in LSI space
                    Shape: (n_components,)
                    
        Raises:
            RuntimeError: if model has not been trained
            ValueError: if query vector has incompatible shape/dtype
            
        Notes:
            - The query is projected into the same space as self.doc_vectors.
            - Result can be compared to self.doc_vectors via cosine similarity.
        """
        if not self.is_trained or self.svd_model is None:
            raise RuntimeError("Model must be trained before transforming queries")
        
        if query_vector is None:
            raise ValueError("Query vector cannot be None or empty")

        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=float)

        if query_vector.size == 0:
            raise ValueError("Query vector cannot be None or empty")

        if query_vector.dtype not in [np.float32, np.float64]:
            query_vector = query_vector.astype(float)

        if query_vector.ndim == 1:
            query_matrix = query_vector.reshape(1, -1)
        elif query_vector.ndim == 2 and query_vector.shape[0] == 1:
            query_matrix = query_vector
        else:
            raise ValueError(
                "Query vector must be 1D or a single-row 2D array"
            )

        expected_shape = self.svd_model.components_.shape[1]
        if query_matrix.shape[1] != expected_shape:
            raise ValueError(
                f"Query vector size {query_matrix.shape[1]} does not match "
                f"expected vocabulary size {expected_shape}"
            )
        
        # Project query: q_lsi = q_tfidf @ V^T
        query_lsi = self.svd_model.transform(query_matrix)[0]
        
        return query_lsi
    
    def save(self, 
             model_path: str = "data/index/lsi_model.pkl",
             vectors_path: str = "data/index/doc_vectors.npy",
             metadata_path: str = "data/index/lsi_metadata.json") -> None:
        """
        Save the trained LSI model and document vectors to disk.
        
        Artifacts:
        - lsi_model.pkl: serialized TruncatedSVD object (scikit-learn model)
        - doc_vectors.npy: semantic document vectors (numpy array)
        - lsi_metadata.json: configuration and metadata
        
        Args:
            model_path (str): path to save SVD model (.pkl)
            vectors_path (str): path to save document vectors (.npy)
            metadata_path (str): path to save metadata (.json)
            
        Raises:
            RuntimeError: if model has not been trained
            IOError: if files cannot be written to disk
            
        Notes:
            - Parent directories are created automatically
            - Uses pickle protocol HIGHEST_PROTOCOL for compatibility
            - Metadata includes configuration for reproducibility
        """
        if not self.is_trained or self.svd_model is None:
            raise RuntimeError("Cannot save untrained model")
        
        try:
            # Save SVD model
            save_pickle(self.svd_model, model_path)
            
            # Save document vectors
            save_numpy(self.doc_vectors, vectors_path)
            
            # Save metadata
            metadata = {
                "n_components": self.n_components,
                "num_documents": self.doc_vectors.shape[0],
                "num_terms": self.svd_model.components_.shape[1],
                "explained_variance_ratio": self.svd_model.explained_variance_ratio_.tolist(),
                "is_trained": self.is_trained
            }
            save_json(metadata, metadata_path)
            
        except IOError as e:
            raise IOError(f"Failed to save LSI model artifacts: {str(e)}") from e
    
    @classmethod
    def load(cls,
             model_path: str = "data/index/lsi_model.pkl",
             vectors_path: str = "data/index/doc_vectors.npy",
             metadata_path: str = "data/index/lsi_metadata.json") -> "LSIModel":
        """
        Load a trained LSI model and document vectors from disk.
        
        Reconstructs a complete LSI model instance from persisted artifacts.
        The model is immediately ready for query projection and similarity
        calculations.
        
        Args:
            model_path (str): path to SVD model (.pkl)
            vectors_path (str): path to document vectors (.npy)
            metadata_path (str): path to metadata (.json)
            
        Returns:
            LSIModel: loaded model instance with trained flag set
            
        Raises:
            FileNotFoundError: if any artifact file is missing
            IOError: if files cannot be read
            pickle.UnpicklingError: if model pickle is corrupted
            
        Notes:
            - All paths must exist; missing files raise FileNotFoundError
            - The returned model is immediately usable for queries
        """
        for path in [model_path, vectors_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing artifact: {path}")
        
        try:
            # Load SVD model
            svd_model = load_pickle(model_path)
            
            # Load document vectors
            doc_vectors = load_numpy(vectors_path)
            
            metadata = None
            if metadata_path and os.path.exists(metadata_path):
                metadata = load_json(metadata_path)

            # Reconstruct LSIModel instance
            n_components = None
            if metadata:
                n_components = metadata.get("n_components")
            if n_components is None:
                n_components = getattr(svd_model, "n_components", None)
            if n_components is None:
                n_components = doc_vectors.shape[1]
            obj = cls(n_components=int(n_components))
            obj.svd_model = svd_model
            obj.doc_vectors = doc_vectors
            obj.is_trained = True
            
            return obj
            
        except (IOError, pickle.UnpicklingError, json.JSONDecodeError) as e:
            raise IOError(f"Failed to load LSI model artifacts: {str(e)}") from e
    
    def get_explained_variance(self) -> np.ndarray:
        """
        Get the explained variance ratio for each latent dimension.
        
        Indicates how much variance (information) each semantic component
        captures from the original TF-IDF space. Useful for:
        - Model quality assessment
        - Determining optimal n_components
        - Understanding semantic structure
        
        Returns:
            ndarray: explained variance ratio per component
                    Shape: (n_components,)
                    Values sum to ≤ 1.0
                    
        Raises:
            RuntimeError: if model has not been trained
        """
        if not self.is_trained or self.svd_model is None:
            raise RuntimeError("Model must be trained before accessing variance")
        
        return self.svd_model.explained_variance_ratio_
    
    def get_singular_values(self) -> np.ndarray:
        """
        Get the singular values from the SVD decomposition.
        
        Singular values represent the importance/magnitude of each latent
        semantic dimension. Larger values indicate dimensions that capture
        more variance in the corpus.
        
        Returns:
            ndarray: singular values in descending order
                    Shape: (n_components,)
                    
        Raises:
            RuntimeError: if model has not been trained
        """
        if not self.is_trained or self.svd_model is None:
            raise RuntimeError("Model must be trained before accessing singular values")
        
        return self.svd_model.singular_values_
    
    def _validate_tfidf_matrix(self, matrix: np.ndarray) -> None:
        """
        Validate TF-IDF matrix format and properties.
        
        Args:
            matrix (ndarray): matrix to validate
            
        Raises:
            ValueError: if matrix is invalid
        """
        if matrix is None:
            raise ValueError("TF-IDF matrix cannot be None")
        
        if not isinstance(matrix, np.ndarray):
            raise ValueError("TF-IDF matrix must be a numpy array")
        
        if matrix.size == 0:
            raise ValueError("TF-IDF matrix cannot be empty")
        
        if len(matrix.shape) != 2:
            raise ValueError(f"TF-IDF matrix must be 2D, got shape {matrix.shape}")
        
        if matrix.dtype not in [np.float32, np.float64]:
            raise ValueError(
                f"TF-IDF matrix must have float dtype, got {matrix.dtype}"
            )
        
        if np.isnan(matrix).any() or np.isinf(matrix).any():
            raise ValueError("TF-IDF matrix contains NaN or Inf values")
    
    def _validate_components_vs_matrix(self, matrix: np.ndarray) -> None:
        """
        Validate that n_components is feasible for the matrix.
        
        Args:
            matrix (ndarray): TF-IDF matrix
            
        Raises:
            ValueError: if n_components is too large
        """
        num_docs, num_terms = matrix.shape
        max_components = min(num_docs, num_terms)
        
        if self.n_components > max_components:
            raise ValueError(
                f"n_components ({self.n_components}) cannot exceed "
                f"min(num_documents={num_docs}, num_terms={num_terms}) = {max_components}"
            )
