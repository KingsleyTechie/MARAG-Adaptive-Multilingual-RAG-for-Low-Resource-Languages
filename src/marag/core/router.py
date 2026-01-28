"""
Adaptive Decision Router for MARAG
Makes intelligent routing decisions based on language profiles and query characteristics
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class RoutingDecision:
    """Container for router decisions with confidence scores"""
    selected_index: str  # 'dense', 'hybrid', 'kg'
    confidence: float
    features_used: Dict[str, float]
    reasoning: str

class AdaptiveRouter(nn.Module):
    """
    Multi-layer perceptron router that selects optimal retrieval strategy
    based on language resources and query features
    """
    
    def __init__(self, input_dim: int, hidden_dims: list = [128, 64]):
        super().__init__()
        self.input_dim = input_dim
        
        # Build sequential layers dynamically
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        # Output layer for 3 strategies
        layers.append(nn.Linear(prev_dim, 3))
        layers.append(nn.Softmax(dim=-1))
        
        self.network = nn.Sequential(*layers)
        
        # Strategy mappings
        self.strategy_names = ['dense', 'hybrid', 'kg']
        self.strategy_descriptions = {
            'dense': 'Dense vector retrieval for languages with good embeddings',
            'hybrid': 'Hybrid sparse-dense for morphologically complex languages',
            'kg': 'Knowledge graph retrieval for very low-resource languages'
        }
    
    def forward(self, language_features: torch.Tensor, query_features: torch.Tensor) -> RoutingDecision:
        """
        Forward pass combining language and query features
        
        Args:
            language_features: Pre-computed language profile vector
            query_features: Real-time query characteristics
        
        Returns:
            RoutingDecision with selected strategy and confidence
        """
        # Combine features
        combined_features = torch.cat([language_features, query_features], dim=-1)
        
        # Ensure correct dimensionality
        if len(combined_features.shape) == 1:
            combined_features = combined_features.unsqueeze(0)
        
        # Get strategy probabilities
        strategy_probs = self.network(combined_features)
        confidence, strategy_idx = torch.max(strategy_probs, dim=-1)
        
        selected_strategy = self.strategy_names[strategy_idx.item()]
        
        # Generate human-readable reasoning
        reasoning = self._generate_reasoning(
            selected_strategy, 
            strategy_probs.detach().cpu().numpy()[0],
            language_features,
            query_features
        )
        
        return RoutingDecision(
            selected_index=selected_strategy,
            confidence=confidence.item(),
            features_used=self._extract_feature_importance(language_features, query_features),
            reasoning=reasoning
        )
    
    def _generate_reasoning(self, strategy: str, probs: np.ndarray, 
                          lang_features: torch.Tensor, query_features: torch.Tensor) -> str:
        """Generate explainable reasoning for routing decision"""
        reasons = []
        
        if strategy == 'dense' and probs[0] > 0.6:
            reasons.append("High embedding space integrity detected")
            if lang_features[0].item() > 0.7:  # Embedding integrity score
                reasons.append("Language has well-structured semantic representations")
        
        elif strategy == 'hybrid' and probs[1] > 0.6:
            reasons.append("Query shows morphological complexity")
            if query_features[1].item() > 0.5:  # Morphological complexity feature
                reasons.append("Benefiting from keyword expansion for agglutinative forms")
        
        elif strategy == 'kg' and probs[2] > 0.6:
            reasons.append("Entity-centric query in low-text resource language")
            if lang_features[2].item() < 0.3:  # Text resource availability
                reasons.append("Leveraging structured knowledge as primary source")
        
        return f"Selected {strategy} retrieval because: " + "; ".join(reasons)
    
    def _extract_feature_importance(self, lang_features: torch.Tensor, 
                                  query_features: torch.Tensor) -> Dict[str, float]:
        """Extract which features contributed most to decision"""
        return {
            'embedding_integrity': lang_features[0].item(),
            'lexical_resources': lang_features[1].item(),
            'kg_density': lang_features[2].item(),
            'query_complexity': query_features[0].item(),
            'entity_density': query_features[1].item()
        }
