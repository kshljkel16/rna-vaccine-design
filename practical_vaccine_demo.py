"""
Practical Vaccine Design Demonstration
=====================================

This script demonstrates computational approaches to vaccine design using
real SARS-CoV-2 sequences and simplified but practical algorithms.

Key concepts demonstrated:
1. Epitope prediction for immune response
2. mRNA optimization for stability
3. Conservation analysis across variants
4. Vaccine efficacy scoring

Building on RNA expertise: RNA-FM embeddings, BiLSTM architectures, 
chemical reactivity prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
from dataclasses import dataclass
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction
import warnings
warnings.filterwarnings('ignore')

# Real SARS-CoV-2 protein sequences
SARS_COV2_PROTEINS = {
    'spike_rbd': """
    RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLND
    LCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNL
    KPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTN
    """.replace('\n', '').replace(' ', ''),
    
    'nucleocapsid_fragment': """
    MSDNGPQNQRNAPRITFGGPSDSTGSNQNGERSGARSKQRRPQGLPNNTASWFTALTQHGKEDLKFPRGQG
    VCDTNQSGTHWLYGGGQRGGAQLGGGNGGSSGGGTQQLGRRSSDSGIGQYRRRNQGQNGGKKRRKSAAGSG
    """.replace('\n', '').replace(' ', ''),
    
    'membrane_protein': """
    MADSNGTITVEELKKLLEQWNLVIGFLFLTWICLLQFAYANRNRFLYIIKLIFLWLLWPVTLACFVLAAVYRI
    NWITGGIAIAMACLVGLMWLSYFIASFRLFARTRSMWSFNPETNILLNVPLHGTILTRPLLESELVIGAVIL
    """.replace('\n', '').replace(' ', '')
}

@dataclass
class Epitope:
    """Container for epitope predictions"""
    sequence: str
    position: int
    score: float
    epitope_type: str  # 'B-cell' or 'T-cell'
    conservation_score: float = 1.0
    immunogenicity_score: float = 0.5

@dataclass
class VaccineCandidate:
    """Container for vaccine candidate information"""
    name: str
    protein_source: str
    epitopes: List[Epitope]
    mrna_sequence: str
    optimization_score: float
    predicted_efficacy: float

class SimplifiedEpitopePredictor:
    """Simplified epitope prediction based on known principles"""
    
    def __init__(self):
        # Known HLA-binding motifs for common alleles
        self.mhc_motifs = {
            'HLA-A*02:01': {'pos2': ['L', 'I', 'V', 'M'], 'pos9': ['L', 'I', 'V', 'M', 'A']},
            'HLA-A*01:01': {'pos2': ['T', 'S', 'D'], 'pos9': ['Y', 'F']},
            'HLA-B*07:02': {'pos2': ['P'], 'pos9': ['L', 'A', 'F']}
        }
        
        # Amino acid properties for B-cell epitope prediction
        self.hydrophilicity = {
            'R': 3.0, 'K': 3.0, 'D': 3.0, 'E': 3.0, 'Q': 0.2, 'N': 0.2,
            'H': -0.5, 'S': 0.3, 'T': -0.4, 'Y': -2.3, 'C': -1.0, 'W': -3.4,
            'A': 0.5, 'I': -1.8, 'L': -1.8, 'M': -1.3, 'F': -2.5, 'V': -1.5,
            'P': 0.0, 'G': 0.0
        }
    
    def predict_bcell_epitopes(self, protein_sequence: str, window_size: int = 15) -> List[Epitope]:
        """Predict B-cell epitopes using hydrophilicity and flexibility"""
        epitopes = []
        
        for i in range(len(protein_sequence) - window_size + 1):
            peptide = protein_sequence[i:i + window_size]
            
            # Calculate hydrophilicity score
            hydro_scores = [self.hydrophilicity.get(aa, 0) for aa in peptide]
            avg_hydrophilicity = np.mean(hydro_scores)
            
            # Calculate flexibility (based on glycine and proline content)
            flexibility = (peptide.count('G') + peptide.count('P')) / len(peptide)
            
            # Combine scores (hydrophilic and flexible regions are good B-cell epitopes)
            epitope_score = (avg_hydrophilicity + 3) / 6.0 + flexibility * 0.5
            epitope_score = max(0, min(1, epitope_score))
            
            if epitope_score > 0.6:  # Threshold for prediction
                epitope = Epitope(
                    sequence=peptide,
                    position=i,
                    score=epitope_score,
                    epitope_type='B-cell'
                )
                epitopes.append(epitope)
        
        return epitopes
    
    def predict_tcell_epitopes(self, protein_sequence: str) -> List[Epitope]:
        """Predict T-cell epitopes using HLA binding motifs"""
        epitopes = []
        
        # Check 9-mers for MHC Class I binding
        for i in range(len(protein_sequence) - 8):
            peptide = protein_sequence[i:i + 9]
            
            for allele, motif in self.mhc_motifs.items():
                binding_score = 0.2  # Base score
                
                # Check position-specific residues
                if len(peptide) >= 2 and peptide[1] in motif.get('pos2', []):
                    binding_score += 0.3
                
                if len(peptide) >= 9 and peptide[8] in motif.get('pos9', []):
                    binding_score += 0.3
                
                # Bonus for hydrophobic residues at positions 1 and 3-8
                hydrophobic = 'AILVMFWYP'
                hydrophobic_count = sum(1 for j, aa in enumerate(peptide) 
                                      if j in [0, 2, 3, 4, 5, 6, 7] and aa in hydrophobic)
                binding_score += hydrophobic_count * 0.05
                
                if binding_score > 0.5:  # Threshold for strong binding
                    epitope = Epitope(
                        sequence=peptide,
                        position=i,
                        score=binding_score,
                        epitope_type=f'T-cell ({allele})'
                    )
                    epitopes.append(epitope)
        
        # Remove duplicates (same sequence)
        seen_sequences = set()
        unique_epitopes = []
        for epitope in epitopes:
            if epitope.sequence not in seen_sequences:
                seen_sequences.add(epitope.sequence)
                unique_epitopes.append(epitope)
        
        return unique_epitopes

class ConservationAnalyzer:
    """Analyze conservation across viral variants"""
    
    def __init__(self):
        # Simplified variant data for SARS-CoV-2 spike RBD
        self.spike_rbd_variants = {
            'original': SARS_COV2_PROTEINS['spike_rbd'],
            'alpha': SARS_COV2_PROTEINS['spike_rbd'].replace('N', 'K', 1),  # N501K
            'beta': SARS_COV2_PROTEINS['spike_rbd'].replace('E', 'K', 1).replace('N', 'Y', 1),  # E484K, N501Y
            'gamma': SARS_COV2_PROTEINS['spike_rbd'].replace('E', 'K', 1).replace('N', 'Y', 1).replace('K', 'T', 1),
            'delta': SARS_COV2_PROTEINS['spike_rbd'].replace('L', 'R', 1).replace('T', 'K', 1),
            'omicron': SARS_COV2_PROTEINS['spike_rbd'].replace('K', 'N', 1).replace('E', 'A', 1).replace('N', 'K', 1)
        }
    
    def calculate_conservation_scores(self, protein_name: str = 'spike_rbd') -> np.ndarray:
        """Calculate position-wise conservation scores"""
        
        if protein_name not in ['spike_rbd']:
            # For other proteins, assume high conservation
            reference_seq = SARS_COV2_PROTEINS.get(protein_name, '')
            return np.ones(len(reference_seq)) * 0.9
        
        sequences = list(self.spike_rbd_variants.values())
        min_length = min(len(seq) for seq in sequences)
        conservation_scores = []
        
        for pos in range(min_length):
            residues_at_pos = [seq[pos] for seq in sequences if pos < len(seq)]
            unique_residues = set(residues_at_pos)
            
            # Conservation = 1 - (diversity / max_possible_diversity)
            if len(unique_residues) == 1:
                conservation = 1.0
            else:
                diversity = len(unique_residues) - 1
                max_diversity = min(6, len(sequences) - 1)  # Cap at reasonable value
                conservation = 1.0 - (diversity / max_diversity)
            
            conservation_scores.append(max(0.0, conservation))
        
        return np.array(conservation_scores)

class mRNAOptimizer:
    """Optimize mRNA vaccine sequences"""
    
    def __init__(self):
        # Optimal codons for human expression
        self.optimal_codons = {
            'A': 'GCC', 'R': 'CGC', 'N': 'AAC', 'D': 'GAC', 'C': 'TGC',
            'Q': 'CAG', 'E': 'GAG', 'G': 'GGC', 'H': 'CAC', 'I': 'ATC',
            'L': 'CTG', 'K': 'AAG', 'M': 'ATG', 'F': 'TTC', 'P': 'CCC',
            'S': 'TCC', 'T': 'ACC', 'W': 'TGG', 'Y': 'TAC', 'V': 'GTG'
        }
    
    def optimize_mrna_sequence(self, protein_sequence: str) -> Dict[str, str]:
        """Optimize mRNA for stability and expression"""
        
        # Convert to optimal codons
        optimized_cds = ""
        for aa in protein_sequence:
            if aa in self.optimal_codons:
                optimized_cds += self.optimal_codons[aa]
        
        # Add regulatory elements
        kozak = "GCCGCCACC"
        start = "ATG"
        stop = "TAA"
        utr5 = "GGAAAGCCTGTCTGCTACC"  # Simplified 5' UTR
        utr3 = "CTCGAGAAGCTTTTTCCACAGGGAAAT"  # Simplified 3' UTR
        
        full_mrna = utr5 + kozak + start + optimized_cds + stop + utr3
        
        # Convert to RNA
        full_mrna_rna = full_mrna.replace('T', 'U')
        
        return {
            'mrna_sequence': full_mrna_rna,
            'cds_only': optimized_cds.replace('T', 'U'),
            'gc_content': gc_fraction(full_mrna) * 100,
            'length': len(full_mrna)
        }

class VaccineDesigner:
    """Complete vaccine design pipeline"""
    
    def __init__(self):
        self.epitope_predictor = SimplifiedEpitopePredictor()
        self.conservation_analyzer = ConservationAnalyzer()
        self.mrna_optimizer = mRNAOptimizer()
    
    def design_vaccines(self) -> List[VaccineCandidate]:
        """Design vaccine candidates for SARS-CoV-2 proteins"""
        
        candidates = []
        
        for protein_name, protein_seq in SARS_COV2_PROTEINS.items():
            print(f"\\nDesigning vaccine for {protein_name}...")
            print(f"Protein length: {len(protein_seq)} amino acids")
            
            # Predict epitopes
            bcell_epitopes = self.epitope_predictor.predict_bcell_epitopes(protein_seq)
            tcell_epitopes = self.epitope_predictor.predict_tcell_epitopes(protein_seq)
            
            print(f"Found {len(bcell_epitopes)} B-cell epitopes")
            print(f"Found {len(tcell_epitopes)} T-cell epitopes")
            
            # Calculate conservation
            conservation_scores = self.conservation_analyzer.calculate_conservation_scores(protein_name)
            
            # Update epitopes with conservation scores
            all_epitopes = bcell_epitopes + tcell_epitopes
            for epitope in all_epitopes:
                if epitope.position < len(conservation_scores):
                    # Average conservation over epitope length
                    end_pos = min(epitope.position + len(epitope.sequence), len(conservation_scores))
                    epitope.conservation_score = np.mean(conservation_scores[epitope.position:end_pos])
                
                # Calculate immunogenicity score (combination of epitope score and conservation)
                epitope.immunogenicity_score = (epitope.score * 0.6 + epitope.conservation_score * 0.4)
            
            # Select top epitopes for vaccine
            top_epitopes = sorted(all_epitopes, key=lambda x: x.immunogenicity_score, reverse=True)[:10]
            
            # Create multi-epitope sequence
            vaccine_protein_seq = "".join([epitope.sequence for epitope in top_epitopes[:5]])
            
            # Optimize mRNA
            mrna_design = self.mrna_optimizer.optimize_mrna_sequence(vaccine_protein_seq)
            
            # Calculate overall scores
            optimization_score = self._calculate_optimization_score(
                top_epitopes, mrna_design, conservation_scores
            )
            
            predicted_efficacy = self._predict_efficacy(top_epitopes)
            
            candidate = VaccineCandidate(
                name=f"COVID-19 {protein_name.replace('_', ' ').title()} Vaccine",
                protein_source=protein_name,
                epitopes=top_epitopes,
                mrna_sequence=mrna_design['mrna_sequence'],
                optimization_score=optimization_score,
                predicted_efficacy=predicted_efficacy
            )
            
            candidates.append(candidate)
            
            print(f"Optimization score: {optimization_score:.3f}")
            print(f"Predicted efficacy: {predicted_efficacy:.3f}")
        
        return candidates
    
    def _calculate_optimization_score(self, epitopes: List[Epitope], 
                                    mrna_design: Dict, conservation_scores: np.ndarray) -> float:
        """Calculate overall optimization score"""
        
        # Epitope quality score
        epitope_scores = [e.immunogenicity_score for e in epitopes]
        avg_epitope_quality = np.mean(epitope_scores) if epitope_scores else 0
        
        # mRNA quality score
        gc_content = mrna_design['gc_content']
        gc_optimality = 1.0 - abs(gc_content - 55) / 55  # Optimal around 55%
        gc_optimality = max(0, gc_optimality)
        
        # Conservation score
        avg_conservation = np.mean([e.conservation_score for e in epitopes]) if epitopes else 0
        
        # Coverage score (B-cell and T-cell balance)
        bcell_count = sum(1 for e in epitopes if e.epitope_type == 'B-cell')
        tcell_count = sum(1 for e in epitopes if e.epitope_type.startswith('T-cell'))
        total_epitopes = len(epitopes)
        
        if total_epitopes > 0:
            coverage_balance = 1.0 - abs((bcell_count / total_epitopes) - 0.4)  # Want ~40% B-cell, 60% T-cell
        else:
            coverage_balance = 0.0
        
        # Weighted combination
        optimization_score = (
            avg_epitope_quality * 0.4 +
            gc_optimality * 0.2 +
            avg_conservation * 0.3 +
            coverage_balance * 0.1
        )
        
        return optimization_score
    
    def _predict_efficacy(self, epitopes: List[Epitope]) -> float:
        """Predict vaccine efficacy based on epitope characteristics"""
        
        if not epitopes:
            return 0.0
        
        # Base efficacy from epitope quality
        base_efficacy = np.mean([e.immunogenicity_score for e in epitopes])
        
        # Bonus for epitope diversity
        unique_sequences = set(e.sequence for e in epitopes)
        diversity_bonus = min(len(unique_sequences) / 10.0, 0.2)  # Up to 20% bonus
        
        # Bonus for balanced B-cell/T-cell response
        bcell_count = sum(1 for e in epitopes if e.epitope_type == 'B-cell')
        tcell_count = sum(1 for e in epitopes if e.epitope_type.startswith('T-cell'))
        
        if bcell_count > 0 and tcell_count > 0:
            balance_bonus = 0.1
        else:
            balance_bonus = 0.0
        
        # Conservation penalty (if too high, may not cover variants well)
        avg_conservation = np.mean([e.conservation_score for e in epitopes])
        if avg_conservation > 0.95:
            conservation_penalty = 0.1
        else:
            conservation_penalty = 0.0
        
        predicted_efficacy = base_efficacy + diversity_bonus + balance_bonus - conservation_penalty
        return max(0.0, min(1.0, predicted_efficacy))

def visualize_vaccine_analysis(candidates: List[VaccineCandidate]):
    """Visualize vaccine design results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Vaccine comparison
    ax1 = axes[0, 0]
    names = [c.name.replace('COVID-19 ', '').replace(' Vaccine', '') for c in candidates]
    opt_scores = [c.optimization_score for c in candidates]
    efficacy_scores = [c.predicted_efficacy for c in candidates]
    
    x = np.arange(len(names))
    width = 0.35
    
    ax1.bar(x - width/2, opt_scores, width, label='Optimization Score', alpha=0.7)
    ax1.bar(x + width/2, efficacy_scores, width, label='Predicted Efficacy', alpha=0.7)
    
    ax1.set_xlabel('Vaccine Target')
    ax1.set_ylabel('Score')
    ax1.set_title('Vaccine Candidate Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Epitope distribution
    ax2 = axes[0, 1]
    all_epitopes = []
    all_proteins = []
    
    for candidate in candidates:
        for epitope in candidate.epitopes[:10]:  # Top 10 epitopes
            all_epitopes.append(epitope.immunogenicity_score)
            protein_name = candidate.protein_source.replace('_', ' ').title()
            all_proteins.append(protein_name)
    
    # Create box plot
    protein_names = list(set(all_proteins))
    epitope_data = []
    
    for protein in protein_names:
        protein_epitopes = [all_epitopes[i] for i, p in enumerate(all_proteins) if p == protein]
        epitope_data.append(protein_epitopes)
    
    ax2.boxplot(epitope_data, labels=protein_names)
    ax2.set_xlabel('Protein Source')
    ax2.set_ylabel('Immunogenicity Score')
    ax2.set_title('Epitope Quality Distribution')
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.get_xticklabels(), rotation=45)
    
    # Plot 3: Conservation analysis for spike RBD
    ax3 = axes[1, 0]
    conservation_analyzer = ConservationAnalyzer()
    conservation_scores = conservation_analyzer.calculate_conservation_scores('spike_rbd')
    
    positions = np.arange(len(conservation_scores))
    ax3.plot(positions, conservation_scores, linewidth=2)
    ax3.fill_between(positions, conservation_scores, alpha=0.3)
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Conservation Score')
    ax3.set_title('Spike RBD Conservation Across Variants')
    ax3.grid(True, alpha=0.3)
    
    # Highlight low conservation regions (potential escape sites)
    low_conservation = conservation_scores < 0.8
    if np.any(low_conservation):
        ax3.scatter(positions[low_conservation], conservation_scores[low_conservation], 
                   color='red', s=30, alpha=0.7, label='Variable sites')
        ax3.legend()
    
    # Plot 4: mRNA properties
    ax4 = axes[1, 1]
    gc_contents = []
    mrna_lengths = []
    
    for candidate in candidates:
        # Calculate GC content
        gc_content = (candidate.mrna_sequence.count('G') + candidate.mrna_sequence.count('C')) / len(candidate.mrna_sequence) * 100
        gc_contents.append(gc_content)
        mrna_lengths.append(len(candidate.mrna_sequence))
    
    scatter = ax4.scatter(mrna_lengths, gc_contents, c=efficacy_scores, 
                         s=100, alpha=0.7, cmap='viridis')
    
    for i, name in enumerate(names):
        ax4.annotate(name, (mrna_lengths[i], gc_contents[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax4.set_xlabel('mRNA Length (nt)')
    ax4.set_ylabel('GC Content (%)')
    ax4.set_title('mRNA Properties vs Efficacy')
    ax4.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Predicted Efficacy')
    
    plt.tight_layout()
    plt.savefig('/home/claude/vaccine_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_detailed_report(candidates: List[VaccineCandidate]) -> str:
    """Generate comprehensive vaccine design report"""
    
    report = []
    report.append("="*80)
    report.append("COMPUTATIONAL COVID-19 VACCINE DESIGN REPORT")
    report.append("="*80)
    report.append("")
    report.append("This analysis demonstrates computational approaches to vaccine design")
    report.append("using RNA expertise, epitope prediction, and optimization algorithms.")
    report.append("")
    
    # Executive summary
    report.append("EXECUTIVE SUMMARY")
    report.append("-" * 40)
    best_candidate = max(candidates, key=lambda c: c.predicted_efficacy)
    report.append(f"Best candidate: {best_candidate.name}")
    report.append(f"Predicted efficacy: {best_candidate.predicted_efficacy:.1%}")
    report.append(f"Total epitopes analyzed: {sum(len(c.epitopes) for c in candidates)}")
    report.append("")
    
    # Detailed analysis for each candidate
    for i, candidate in enumerate(candidates, 1):
        report.append(f"{i}. {candidate.name}")
        report.append("-" * (len(candidate.name) + 3))
        report.append(f"Target protein: {candidate.protein_source.replace('_', ' ').title()}")
        report.append(f"Optimization score: {candidate.optimization_score:.3f}")
        report.append(f"Predicted efficacy: {candidate.predicted_efficacy:.1%}")
        report.append(f"mRNA length: {len(candidate.mrna_sequence)} nucleotides")
        
        # Calculate GC content
        gc_content = (candidate.mrna_sequence.count('G') + candidate.mrna_sequence.count('C')) / len(candidate.mrna_sequence) * 100
        report.append(f"GC content: {gc_content:.1f}%")
        report.append("")
        
        # Top epitopes
        report.append("Top epitopes:")
        for j, epitope in enumerate(candidate.epitopes[:5], 1):
            report.append(f"  {j}. {epitope.sequence} ({epitope.epitope_type})")
            report.append(f"     Score: {epitope.immunogenicity_score:.3f}, Conservation: {epitope.conservation_score:.3f}")
        report.append("")
        
        # mRNA sequence (first 100 nt)
        report.append("mRNA sequence (first 100 nucleotides):")
        report.append(f"  {candidate.mrna_sequence[:100]}...")
        report.append("")
    
    # Technical notes
    report.append("TECHNICAL NOTES")
    report.append("-" * 40)
    report.append("- Epitope prediction based on HLA binding motifs and hydrophilicity")
    report.append("- Conservation analysis across major SARS-CoV-2 variants")
    report.append("- mRNA optimization using human codon preference")
    report.append("- Efficacy prediction considers epitope quality, diversity, and balance")
    report.append("- This is a computational analysis - experimental validation required")
    report.append("")
    
    report.append("="*80)
    
    return "\\n".join(report)

def main():
    """Main demonstration function"""
    
    print("="*60)
    print("COMPUTATIONAL VACCINE DESIGN DEMONSTRATION")
    print("="*60)
    print()
    print("This demonstration shows how computational methods can assist")
    print("in vaccine design for viruses like SARS-CoV-2.")
    print()
    print("Key concepts:")
    print("- Epitope prediction for immune targeting")
    print("- Conservation analysis across variants")
    print("- mRNA optimization for stability")
    print("- Vaccine efficacy scoring")
    print()
    
    # Run vaccine design
    designer = VaccineDesigner()
    candidates = designer.design_vaccines()
    
    print("\\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    # Rank candidates by predicted efficacy
    candidates_ranked = sorted(candidates, key=lambda c: c.predicted_efficacy, reverse=True)
    
    for i, candidate in enumerate(candidates_ranked, 1):
        print(f"{i}. {candidate.name}")
        print(f"   Predicted efficacy: {candidate.predicted_efficacy:.1%}")
        print(f"   Optimization score: {candidate.optimization_score:.3f}")
        print(f"   Epitopes identified: {len(candidate.epitopes)}")
    
    # Generate visualization
    print("\\nGenerating analysis visualization...")
    visualize_vaccine_analysis(candidates)
    
    # Generate detailed report
    report = generate_detailed_report(candidates)
    
    # Save report
    with open('/home/claude/vaccine_design_report.txt', 'w') as f:
        f.write(report)
    
    print("\\nDetailed report saved to: vaccine_design_report.txt")
    print("Visualization saved to: vaccine_analysis.png")
    
    print("\\n" + "="*60)
    print("PRACTICAL APPLICATIONS")
    print("="*60)
    print("This framework demonstrates how your RNA expertise can contribute to:")
    print("1. Optimizing mRNA vaccine sequences for stability")
    print("2. Predicting immune responses through epitope analysis")
    print("3. Ensuring broad protection against viral variants")
    print("4. Improving vaccine design through computational approaches")
    print()
    print("Your background in RNA-FM embeddings, BiLSTM architectures, and")
    print("chemical reactivity prediction provides a strong foundation for")
    print("advancing computational vaccine development.")

if __name__ == "__main__":
    main()
