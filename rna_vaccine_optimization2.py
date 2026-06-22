"""
RNA Vaccine Optimization using Chemical Reactivity Prediction
============================================================

This script builds on RNA chemical reactivity prediction research to optimize
mRNA vaccine sequences for stability, translation efficiency, and reduced
immunogenicity of the RNA itself.

Based on research experience with:
- RNA-FM embeddings for sequence representation
- BiLSTM and CNN+Attention architectures 
- DMS, 1M7, 2A3 chemical probing data
- RNA secondary structure prediction

Author: Building on RNA reactivity prediction thesis work
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import re
from dataclasses import dataclass
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction
import warnings
warnings.filterwarnings('ignore')

@dataclass
class RNAReactivityPrediction:
    """Container for RNA reactivity predictions"""
    sequence: str
    dms_reactivity: np.ndarray
    m7_reactivity: np.ndarray
    a3_reactivity: np.ndarray
    structure_prob: np.ndarray
    stability_score: float

class RNAEmbeddingModel:
    """
    Simplified RNA embedding model inspired by RNA-FM
    In practice, you would use the actual RNA-FM model
    """
    
    def __init__(self, embedding_dim: int = 640):
        self.embedding_dim = embedding_dim
        self.nucleotide_to_id = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}
        
        # Simple position-wise embedding (placeholder for RNA-FM)
        self.position_embeddings = np.random.randn(1000, embedding_dim) * 0.1
        self.nucleotide_embeddings = np.random.randn(5, embedding_dim) * 0.1
    
    def encode_sequence(self, sequence: str) -> np.ndarray:
        """Convert RNA sequence to embeddings"""
        sequence = sequence.replace('T', 'U')  # Convert DNA to RNA
        
        embeddings = []
        for i, nucleotide in enumerate(sequence):
            nuc_id = self.nucleotide_to_id.get(nucleotide, 4)  # 4 for unknown
            
            # Combine nucleotide and position embeddings
            nuc_emb = self.nucleotide_embeddings[nuc_id]
            pos_emb = self.position_embeddings[i % len(self.position_embeddings)]
            
            combined = nuc_emb + pos_emb
            embeddings.append(combined)
        
        return np.array(embeddings)

class RNAReactivityPredictor(nn.Module):
    """
    BiLSTM model for predicting RNA chemical reactivity
    Based on successful architecture from thesis research
    """
    
    def __init__(self, input_dim: int = 640, hidden_dim: int = 256, 
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # BiLSTM layers (found to work best in research)
        self.bilstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            hidden_dim * 2, num_heads=8, dropout=dropout, batch_first=True
        )
        
        # Output heads for different chemical probes
        self.dms_head = nn.Linear(hidden_dim * 2, 1)
        self.m7_head = nn.Linear(hidden_dim * 2, 1)
        self.a3_head = nn.Linear(hidden_dim * 2, 1)
        
        # Structure probability head
        self.structure_head = nn.Linear(hidden_dim * 2, 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the model"""
        
        # BiLSTM encoding
        lstm_out, _ = self.bilstm(x)
        lstm_out = self.dropout(lstm_out)
        
        # Self-attention
        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attended_out = self.dropout(attended_out)
        
        # Predictions for different chemical probes
        dms_pred = torch.sigmoid(self.dms_head(attended_out))
        m7_pred = torch.sigmoid(self.m7_head(attended_out))
        a3_pred = torch.sigmoid(self.a3_head(attended_out))
        
        # Structure probability (paired vs unpaired)
        structure_pred = torch.sigmoid(self.structure_head(attended_out))
        
        return {
            'dms': dms_pred.squeeze(-1),
            'm7': m7_pred.squeeze(-1),
            'a3': a3_pred.squeeze(-1),
            'structure': structure_pred.squeeze(-1)
        }

class mRNAOptimizer:
    """
    Optimize mRNA vaccine sequences based on chemical reactivity predictions
    """
    
    def __init__(self, reactivity_model_path: Optional[str] = None):
        self.embedding_model = RNAEmbeddingModel()
        
        # Load pre-trained reactivity model or create new one
        if reactivity_model_path:
            self.reactivity_model = torch.load(reactivity_model_path)
        else:
            self.reactivity_model = RNAReactivityPredictor()
            # In practice, you would train this model on DMS, 1M7, 2A3 data
        
        self.reactivity_model.eval()
        
        # Human codon usage for optimization
        self.human_codon_usage = {
            'UUU': 17.6, 'UUC': 20.3, 'UUA': 7.7, 'UUG': 12.9,
            'UCU': 15.2, 'UCC': 17.7, 'UCA': 12.2, 'UCG': 4.4,
            'UAU': 12.2, 'UAC': 15.3, 'UAA': 1.0, 'UAG': 0.8,
            'UGU': 10.6, 'UGC': 12.6, 'UGA': 1.6, 'UGG': 13.2,
            'CUU': 13.2, 'CUC': 19.6, 'CUA': 7.2, 'CUG': 39.6,
            'CCU': 17.5, 'CCC': 19.8, 'CCA': 16.9, 'CCG': 6.9,
            'CAU': 10.9, 'CAC': 15.1, 'CAA': 12.3, 'CAG': 34.2,
            'CGU': 4.5, 'CGC': 10.4, 'CGA': 6.2, 'CGG': 11.4,
            'AUU': 16.0, 'AUC': 20.8, 'AUA': 7.5, 'AUG': 22.0,
            'ACU': 13.1, 'ACC': 18.9, 'ACA': 15.1, 'ACG': 6.1,
            'AAU': 17.0, 'AAC': 19.1, 'AAA': 24.4, 'AAG': 31.9,
            'AGU': 12.1, 'AGC': 19.5, 'AGA': 12.2, 'AGG': 12.0,
            'GUU': 11.0, 'GUC': 14.5, 'GUA': 7.1, 'GUG': 28.1,
            'GCU': 18.4, 'GCC': 27.7, 'GCA': 15.8, 'GCG': 7.4,
            'GAU': 21.8, 'GAC': 25.1, 'GAA': 29.0, 'GAG': 39.6,
            'GGU': 10.8, 'GGC': 22.2, 'GGA': 16.5, 'GGG': 16.5
        }
    
    def predict_rna_reactivity(self, rna_sequence: str) -> RNAReactivityPrediction:
        """Predict chemical reactivity for RNA sequence"""
        
        # Encode sequence
        embeddings = self.embedding_model.encode_sequence(rna_sequence)
        
        # Convert to tensor
        x = torch.FloatTensor(embeddings).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            predictions = self.reactivity_model(x)
        
        # Extract predictions
        dms_reactivity = predictions['dms'][0].numpy()
        m7_reactivity = predictions['m7'][0].numpy()
        a3_reactivity = predictions['a3'][0].numpy()
        structure_prob = predictions['structure'][0].numpy()
        
        # Calculate overall stability score
        stability_score = self._calculate_stability_score(
            dms_reactivity, m7_reactivity, a3_reactivity, structure_prob
        )
        
        return RNAReactivityPrediction(
            sequence=rna_sequence,
            dms_reactivity=dms_reactivity,
            m7_reactivity=m7_reactivity,
            a3_reactivity=a3_reactivity,
            structure_prob=structure_prob,
            stability_score=stability_score
        )
    
    def optimize_vaccine_sequence(self, protein_sequence: str, 
                                num_iterations: int = 100) -> Dict[str, str]:
        """
        Optimize mRNA vaccine sequence for stability and efficacy
        """
        
        # Start with optimal codon usage
        current_rna = self._optimize_codon_usage(protein_sequence)
        current_rna = self._add_regulatory_elements(current_rna)
        
        best_rna = current_rna
        best_score = -float('inf')
        
        scores_history = []
        
        for iteration in range(num_iterations):
            # Generate variants with synonymous codon substitutions
            variants = self._generate_synonymous_variants(current_rna, num_variants=10)
            
            # Evaluate each variant
            for variant in variants:
                reactivity_pred = self.predict_rna_reactivity(variant)
                
                # Calculate optimization score
                opt_score = self._calculate_optimization_score(
                    reactivity_pred, variant
                )
                
                if opt_score > best_score:
                    best_score = opt_score
                    best_rna = variant
                    current_rna = variant
            
            scores_history.append(best_score)
            
            if iteration % 20 == 0:
                print(f"Iteration {iteration}: Best score = {best_score:.4f}")
        
        # Final analysis
        final_prediction = self.predict_rna_reactivity(best_rna)
        
        return {
            'optimized_sequence': best_rna,
            'optimization_score': best_score,
            'stability_score': final_prediction.stability_score,
            'gc_content': gc_fraction(best_rna) * 100,
            'length': len(best_rna),
            'scores_history': scores_history
        }
    
    def analyze_vaccine_rna_properties(self, rna_sequence: str) -> Dict:
        """Comprehensive analysis of vaccine RNA properties"""
        
        # Get reactivity predictions
        reactivity_pred = self.predict_rna_reactivity(rna_sequence)
        
        # Basic sequence properties
        gc_content = gc_fraction(rna_sequence) * 100
        length = len(rna_sequence)
        
        # Find problematic regions
        high_reactivity_regions = self._find_high_reactivity_regions(reactivity_pred)
        unstable_regions = self._find_unstable_regions(reactivity_pred)
        
        # Secondary structure analysis
        structured_regions = np.where(reactivity_pred.structure_prob > 0.7)[0]
        unstructured_regions = np.where(reactivity_pred.structure_prob < 0.3)[0]
        
        # Calculate regional stability
        utr5_stability = self._calculate_regional_stability(reactivity_pred, 0, 100)
        cds_stability = self._calculate_regional_stability(reactivity_pred, 100, -100)
        utr3_stability = self._calculate_regional_stability(reactivity_pred, -100, -1)
        
        return {
            'reactivity_prediction': reactivity_pred,
            'gc_content': gc_content,
            'length': length,
            'high_reactivity_regions': high_reactivity_regions,
            'unstable_regions': unstable_regions,
            'structured_positions': structured_regions,
            'unstructured_positions': unstructured_regions,
            'utr5_stability': utr5_stability,
            'cds_stability': cds_stability,
            'utr3_stability': utr3_stability,
            'overall_stability': reactivity_pred.stability_score
        }
    
    def _optimize_codon_usage(self, protein_sequence: str) -> str:
        """Optimize codon usage for human expression"""
        
        # Genetic code table (DNA -> RNA)
        optimal_codons = {
            'F': 'UUC', 'L': 'CUG', 'S': 'UCC', 'Y': 'UAC',
            'C': 'UGC', 'W': 'UGG', 'P': 'CCC', 'H': 'CAC',
            'Q': 'CAG', 'R': 'CGC', 'I': 'AUC', 'T': 'ACC',
            'N': 'AAC', 'K': 'AAG', 'V': 'GUG', 'A': 'GCC',
            'D': 'GAC', 'E': 'GAG', 'G': 'GGC', 'M': 'AUG'
        }
        
        rna_sequence = ""
        for aa in protein_sequence:
            if aa in optimal_codons:
                rna_sequence += optimal_codons[aa]
            elif aa == '*':
                rna_sequence += 'UAA'  # Stop codon
        
        return rna_sequence
    
    def _add_regulatory_elements(self, cds_sequence: str) -> str:
        """Add 5' and 3' UTRs and other regulatory elements"""
        
        # 5' UTR with strong Kozak sequence
        utr5 = "GGAAAGCCUGUCUCUGCUACC"
        kozak = "GCCGCCACC"
        start_codon = "AUG"
        
        # 3' UTR for stability
        utr3 = "CUCGAGAAGCUUUUUCCACAGGGAAAU"
        
        # Poly(A) tail (simplified)
        poly_a = "A" * 30
        
        full_sequence = utr5 + kozak + start_codon + cds_sequence + "UAA" + utr3 + poly_a
        
        return full_sequence
    
    def _generate_synonymous_variants(self, rna_sequence: str, 
                                    num_variants: int = 10) -> List[str]:
        """Generate synonymous variants using alternative codons"""
        
        variants = []
        
        # Define synonymous codon groups
        synonymous_codons = {
            'UUU': ['UUC'], 'UUC': ['UUU'],
            'UUA': ['UUG', 'CUU', 'CUC', 'CUA', 'CUG'],
            'UUG': ['UUA', 'CUU', 'CUC', 'CUA', 'CUG'],
            'UCU': ['UCC', 'UCA', 'UCG', 'AGU', 'AGC'],
            'UCC': ['UCU', 'UCA', 'UCG', 'AGU', 'AGC'],
            # Add more synonymous groups...
        }
        
        for _ in range(num_variants):
            variant = rna_sequence
            
            # Make random synonymous substitutions
            for _ in range(np.random.randint(1, 10)):
                # Find random codon position
                codon_start = np.random.randint(0, len(variant) - 2)
                codon_start = codon_start - (codon_start % 3)  # Align to codon
                
                if codon_start + 3 <= len(variant):
                    current_codon = variant[codon_start:codon_start + 3]
                    
                    if current_codon in synonymous_codons:
                        alternatives = synonymous_codons[current_codon]
                        if alternatives:
                            new_codon = np.random.choice(alternatives)
                            variant = (variant[:codon_start] + 
                                     new_codon + 
                                     variant[codon_start + 3:])
            
            variants.append(variant)
        
        return variants
    
    def _calculate_stability_score(self, dms_reactivity: np.ndarray,
                                 m7_reactivity: np.ndarray,
                                 a3_reactivity: np.ndarray,
                                 structure_prob: np.ndarray) -> float:
        """Calculate overall RNA stability score"""
        
        # Lower reactivity generally indicates more stable structure
        dms_stability = 1.0 - np.mean(dms_reactivity)
        m7_stability = 1.0 - np.mean(m7_reactivity)
        a3_stability = 1.0 - np.mean(a3_reactivity)
        
        # Moderate structure is optimal (not too tight, not too loose)
        structure_optimality = 1.0 - np.abs(np.mean(structure_prob) - 0.6)
        
        # Weighted combination
        stability_score = (dms_stability * 0.3 + 
                          m7_stability * 0.3 + 
                          a3_stability * 0.2 + 
                          structure_optimality * 0.2)
        
        return stability_score
    
    def _calculate_optimization_score(self, reactivity_pred: RNAReactivityPrediction,
                                    rna_sequence: str) -> float:
        """Calculate overall optimization score for vaccine RNA"""
        
        # Stability component
        stability_weight = 0.4
        stability_component = reactivity_pred.stability_score * stability_weight
        
        # GC content optimization (50-60% is optimal)
        gc_content = gc_fraction(rna_sequence) * 100
        gc_optimality = 1.0 - abs(gc_content - 55) / 55
        gc_component = gc_optimality * 0.2
        
        # Avoid problematic sequences
        problem_penalty = 0.0
        
        # Penalize poly-A/U runs that can cause termination
        if 'AAAA' in rna_sequence:
            problem_penalty += 0.1
        if 'UUUU' in rna_sequence:
            problem_penalty += 0.1
        
        # Penalize very high reactivity regions (unstable)
        high_reactivity_fraction = np.mean(reactivity_pred.dms_reactivity > 0.8)
        if high_reactivity_fraction > 0.1:
            problem_penalty += high_reactivity_fraction * 0.2
        
        # Translation efficiency (prefer moderate structure in 5' UTR)
        utr5_structure = np.mean(reactivity_pred.structure_prob[:100])  # First 100 nt
        translation_optimality = 1.0 - abs(utr5_structure - 0.4)  # Want some but not too much structure
        translation_component = translation_optimality * 0.2
        
        # Immunogenicity reduction (avoid highly structured regions that may trigger RIG-I)
        immunogenicity_penalty = np.mean(reactivity_pred.structure_prob > 0.9) * 0.2
        
        total_score = (stability_component + 
                      gc_component + 
                      translation_component - 
                      problem_penalty - 
                      immunogenicity_penalty)
        
        return total_score
    
    def _find_high_reactivity_regions(self, reactivity_pred: RNAReactivityPrediction,
                                    threshold: float = 0.8) -> List[Tuple[int, int]]:
        """Find regions of high chemical reactivity"""
        
        # Combine all reactivity measures
        combined_reactivity = (reactivity_pred.dms_reactivity + 
                             reactivity_pred.m7_reactivity + 
                             reactivity_pred.a3_reactivity) / 3.0
        
        high_regions = []
        in_region = False
        region_start = 0
        
        for i, reactivity in enumerate(combined_reactivity):
            if reactivity > threshold and not in_region:
                region_start = i
                in_region = True
            elif reactivity <= threshold and in_region:
                high_regions.append((region_start, i))
                in_region = False
        
        if in_region:
            high_regions.append((region_start, len(combined_reactivity)))
        
        return high_regions
    
    def _find_unstable_regions(self, reactivity_pred: RNAReactivityPrediction) -> List[Tuple[int, int]]:
        """Find regions with predicted instability"""
        
        # Regions with very high DMS reactivity and low structure probability
        unstable_mask = ((reactivity_pred.dms_reactivity > 0.7) & 
                        (reactivity_pred.structure_prob < 0.3))
        
        unstable_regions = []
        in_region = False
        region_start = 0
        
        for i, is_unstable in enumerate(unstable_mask):
            if is_unstable and not in_region:
                region_start = i
                in_region = True
            elif not is_unstable and in_region:
                unstable_regions.append((region_start, i))
                in_region = False
        
        if in_region:
            unstable_regions.append((region_start, len(unstable_mask)))
        
        return unstable_regions
    
    def _calculate_regional_stability(self, reactivity_pred: RNAReactivityPrediction,
                                    start: int, end: int) -> float:
        """Calculate stability score for a specific region"""
        
        if end == -1:
            end = len(reactivity_pred.sequence)
        if start < 0:
            start = len(reactivity_pred.sequence) + start
        if end < 0:
            end = len(reactivity_pred.sequence) + end
        
        start = max(0, start)
        end = min(len(reactivity_pred.sequence), end)
        
        if start >= end:
            return 0.0
        
        region_dms = reactivity_pred.dms_reactivity[start:end]
        region_structure = reactivity_pred.structure_prob[start:end]
        
        stability = (1.0 - np.mean(region_dms)) * 0.7 + np.mean(region_structure) * 0.3
        
        return stability

def visualize_rna_analysis(analysis_results: Dict, save_path: Optional[str] = None):
    """Visualize RNA analysis results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    reactivity_pred = analysis_results['reactivity_prediction']
    sequence_length = len(reactivity_pred.sequence)
    positions = np.arange(sequence_length)
    
    # Plot 1: Chemical reactivity profiles
    ax1 = axes[0, 0]
    ax1.plot(positions, reactivity_pred.dms_reactivity, label='DMS', alpha=0.7)
    ax1.plot(positions, reactivity_pred.m7_reactivity, label='1M7', alpha=0.7)
    ax1.plot(positions, reactivity_pred.a3_reactivity, label='2A3', alpha=0.7)
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Reactivity')
    ax1.set_title('Chemical Reactivity Profiles')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Structure probability
    ax2 = axes[0, 1]
    ax2.plot(positions, reactivity_pred.structure_prob, color='purple', alpha=0.7)
    ax2.fill_between(positions, reactivity_pred.structure_prob, alpha=0.3, color='purple')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Structure Probability')
    ax2.set_title('Predicted Secondary Structure')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Problem regions
    ax3 = axes[1, 0]
    
    # Mark high reactivity regions
    for start, end in analysis_results['high_reactivity_regions']:
        ax3.axvspan(start, end, alpha=0.3, color='red', label='High Reactivity')
    
    # Mark unstable regions
    for start, end in analysis_results['unstable_regions']:
        ax3.axvspan(start, end, alpha=0.3, color='orange', label='Unstable')
    
    ax3.plot(positions, reactivity_pred.dms_reactivity, color='black', alpha=0.5)
    ax3.set_xlabel('Position')
    ax3.set_ylabel('DMS Reactivity')
    ax3.set_title('Problem Regions')
    
    # Remove duplicate labels
    handles, labels = ax3.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax3.legend(by_label.values(), by_label.keys())
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Regional stability scores
    ax4 = axes[1, 1]
    regions = ['5\' UTR', 'CDS', '3\' UTR']
    stability_scores = [
        analysis_results['utr5_stability'],
        analysis_results['cds_stability'],
        analysis_results['utr3_stability']
    ]
    
    bars = ax4.bar(regions, stability_scores, color=['lightblue', 'lightgreen', 'lightcoral'])
    ax4.set_ylabel('Stability Score')
    ax4.set_title('Regional Stability Analysis')
    ax4.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, score in zip(bars, stability_scores):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def demo_mrna_vaccine_optimization():
    """Demonstration of mRNA vaccine optimization"""
    
    # Example spike protein RBD sequence (SARS-CoV-2)
    spike_rbd_sequence = """
    RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT
    """.replace('\n', '').replace(' ', '')
    
    # Initialize optimizer
    optimizer = mRNAOptimizer()
    
    print("Starting mRNA vaccine optimization...")
    print(f"Protein sequence length: {len(spike_rbd_sequence)} amino acids")
    
    # Optimize the sequence
    optimization_results = optimizer.optimize_vaccine_sequence(
        spike_rbd_sequence, num_iterations=50
    )
    
    print(f"\nOptimization completed!")
    print(f"Final optimization score: {optimization_results['optimization_score']:.4f}")
    print(f"Stability score: {optimization_results['stability_score']:.4f}")
    print(f"GC content: {optimization_results['gc_content']:.1f}%")
    print(f"mRNA length: {optimization_results['length']} nucleotides")
    
    # Analyze the optimized sequence
    analysis_results = optimizer.analyze_vaccine_rna_properties(
        optimization_results['optimized_sequence']
    )
    
    print(f"\nDetailed Analysis:")
    print(f"High reactivity regions: {len(analysis_results['high_reactivity_regions'])}")
    print(f"Unstable regions: {len(analysis_results['unstable_regions'])}")
    print(f"5' UTR stability: {analysis_results['utr5_stability']:.3f}")
    print(f"CDS stability: {analysis_results['cds_stability']:.3f}")
    print(f"3' UTR stability: {analysis_results['utr3_stability']:.3f}")
    
    # Visualize results
    print("\nGenerating visualization...")
    visualize_rna_analysis(analysis_results, 'rna_analysis.png')
    
    # Plot optimization progress
    plt.figure(figsize=(10, 6))
    plt.plot(optimization_results['scores_history'])
    plt.xlabel('Iteration')
    plt.ylabel('Optimization Score')
    plt.title('mRNA Vaccine Optimization Progress')
    plt.grid(True, alpha=0.3)
    plt.savefig('optimization_progress.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return optimization_results, analysis_results

if __name__ == "__main__":
    # Run the demonstration
    opt_results, analysis_results = demo_mrna_vaccine_optimization()
    
    print(f"\nOptimized mRNA sequence (first 200 nt):")
    print(opt_results['optimized_sequence'][:200] + "...")
