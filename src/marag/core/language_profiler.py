"""
Language Resource Profiler for MARAG
Computes quantitative profiles for each supported language
"""

import json
import numpy as np
from typing import List, Dict
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class LanguageProfile:
    """Comprehensive profile of a language's resources"""
    language_code: str
    language_name: str
    
    # Core metrics (0-1 scale)
    embedding_integrity: float
    lexical_resources: float
    structured_knowledge_density: float
    translation_confidence: float
    
    # Derived features
    overall_resource_score: float
    recommended_strategy: str
    
    # Metadata
    corpus_size: int
    unique_entities: int
    evaluation_date: str
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert profile to feature vector for router"""
        return np.array([
            self.embedding_integrity,
            self.lexical_resources,
            self.structured_knowledge_density,
            self.translation_confidence,
            self.overall_resource_score
        ])
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return asdict(self)

class LanguageProfiler:
    """Main class for computing and managing language profiles"""
    
    def __init__(self, data_dir: Path = Path("data/language_profiles")):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def compute_profile(self, language_code: str, 
                       corpus_path: Path,
                       evaluation_data: Dict) -> LanguageProfile:
        """
        Compute comprehensive language profile
        
        Args:
            language_code: ISO 639-1 language code
            corpus_path: Path to language corpus
            evaluation_data: Parallel data for embedding evaluation
        
        Returns:
            Complete LanguageProfile object
        """
        
        # Load and analyze corpus
        corpus_stats = self._analyze_corpus(corpus_path)
        
        # Compute embedding integrity using parallel data
        embedding_score = self._compute_embedding_integrity(
            evaluation_data["source_sentences"],
            evaluation_data["target_sentences"]
        )
        
        # Calculate lexical resource score
        lexical_score = self._compute_lexical_resources(corpus_stats)
        
        # Estimate knowledge graph density
        kg_density = self._estimate_kg_density(language_code)
        
        # Machine translation confidence (if available)
        mt_confidence = evaluation_data.get("translation_bleu", 0.3)
        
        # Overall resource score (weighted average)
        overall_score = (
            0.3 * embedding_score +
            0.3 * lexical_score +
            0.25 * kg_density +
            0.15 * mt_confidence
        )
        
        # Recommend strategy based on scores
        if kg_density > 0.6 and lexical_score < 0.4:
            recommended = "kg"
        elif embedding_score > 0.7:
            recommended = "dense"
        else:
            recommended = "hybrid"
        
        profile = LanguageProfile(
            language_code=language_code,
            language_name=self._get_language_name(language_code),
            embedding_integrity=round(embedding_score, 3),
            lexical_resources=round(lexical_score, 3),
            structured_knowledge_density=round(kg_density, 3),
            translation_confidence=round(mt_confidence, 3),
            overall_resource_score=round(overall_score, 3),
            recommended_strategy=recommended,
            corpus_size=corpus_stats["total_documents"],
            unique_entities=corpus_stats["unique_entities"],
            evaluation_date=str(datetime.now().date())
        )
        
        # Save profile
        self._save_profile(profile)
        
        return profile
    
    def _analyze_corpus(self, corpus_path: Path) -> Dict:
        """Analyze corpus statistics"""
        # Implementation for corpus analysis
        pass
    
    def _compute_embedding_integrity(self, source_sents: List[str], 
                                   target_sents: List[str]) -> float:
        """Compute how well embeddings preserve semantic similarity"""
        # Implementation using sentence transformers
        pass
    
    def _save_profile(self, profile: LanguageProfile):
        """Save profile to JSON file"""
        profile_path = self.data_dir / f"{profile.language_code}_profile.json"
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(profile.to_dict(), f, indent=2, ensure_ascii=False)
