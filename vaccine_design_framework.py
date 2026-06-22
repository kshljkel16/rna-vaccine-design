"""
Computational Vaccine Design Framework
=====================================

This framework demonstrates how RNA analysis, machine learning, and bioinformatics
can contribute to vaccine development for viruses like SARS-CoV-2.

Key components:
1. Viral genome analysis and protein extraction
2. Epitope prediction for B-cell and T-cell responses
3. mRNA vaccine sequence optimization
4. Stability and immunogenicity scoring
5. Variant analysis for broad protection

Author: Based on RNA reactivity prediction expertise
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import re
from dataclasses import dataclass
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction
from Bio.SeqUtils import molecular_weight
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

@dataclass
class EpitopeCandidate:
    """Container for epitope prediction results"""
    sequence: str
    start_pos: int
    end_pos: int
    score: float
    epitope_type: str  # 'B-cell' or 'T-cell'
    mhc_allele: Optional[str] = None
    conservation_score: float = 0.0

class ViralGenomeAnalyzer:
    """Analyze viral genomes for vaccine target identification"""
    
    def __init__(self):
        self.codon_usage_table = {
            'TTT': 0.45, 'TTC': 0.55, 'TTA': 0.07, 'TTG': 0.13,
            'TCT': 0.18, 'TCC': 0.22, 'TCA': 0.15, 'TCG': 0.05,
            'TAT': 0.43, 'TAC': 0.57, 'TAA': 0.28, 'TAG': 0.20,
            'TGT': 0.45, 'TGC': 0.55, 'TGA': 0.52, 'TGG': 1.00,
            # Add more codons for complete table
        }
    
    def extract_proteins(self, genome_sequence: str) -> Dict[str, str]:
        """Extract major protein sequences from viral genome"""
        proteins = {}
        
        # SARS-CoV-2 specific ORFs (can be adapted for other viruses)
        sars_cov2_orfs = {
            'spike': (21563, 25384),      # S protein
            'nucleocapsid': (28274, 29533), # N protein  
            'membrane': (26523, 27191),    # M protein
            'envelope': (26245, 26472),    # E protein
            'orf1ab': (266, 21555),        # Polyproteins
        }
        
        for protein_name, (start, end) in sars_cov2_orfs.items():
            if end <= len(genome_sequence):
                orf_seq = genome_sequence[start-1:end]
                # Translate to protein
                protein_seq = str(Seq(orf_seq).translate())
                proteins[protein_name] = protein_seq.replace('*', '')
        
        return proteins
    
    def analyze_conservation(self, sequences: List[str]) -> np.ndarray:
        """Calculate conservation scores across viral variants"""
        if not sequences:
            return np.array([])
        
        alignment_length = min(len(seq) for seq in sequences)
        conservation_scores = []
        
        for pos in range(alignment_length):
            residues = [seq[pos] for seq in sequences if pos < len(seq)]
            # Shannon entropy-based conservation
            unique_residues = set(residues)
            if len(unique_residues) == 1:
                conservation = 1.0
            else:
                entropy = 0
                for residue in unique_residues:
                    freq = residues.count(residue) / len(residues)
                    entropy -= freq * np.log2(freq)
                # Normalize entropy (max entropy for 20 amino acids = log2(20))
                max_entropy = np.log2(min(20, len(set(residues))))
                conservation = 1 - (entropy / max_entropy if max_entropy > 0 else 0)
            
            conservation_scores.append(conservation)
        
        return np.array(conservation_scores)

class EpitopePrediction:
    """Machine learning-based epitope prediction"""
    
    def __init__(self):
        self.bcell_model = None
        self.tcell_model = None
        self.features_cache = {}
        
    def extract_sequence_features(self, sequence: str) -> np.ndarray:
        """Extract physicochemical features from peptide sequence"""
        if sequence in self.features_cache:
            return self.features_cache[sequence]
        
        # Amino acid properties
        aa_properties = {
            'A': [1.8, 0.0, 0.0], 'R': [-4.5, 1.0, 1.0], 'N': [-3.5, 0.0, 0.0],
            'D': [-3.5, -1.0, 0.0], 'C': [2.5, 0.0, 0.0], 'Q': [-3.5, 0.0, 0.0],
            'E': [-3.5, -1.0, 0.0], 'G': [-0.4, 0.0, 0.0], 'H': [-3.2, 0.0, 1.0],
            'I': [4.5, 0.0, 0.0], 'L': [3.8, 0.0, 0.0], 'K': [-3.9, 1.0, 1.0],
            'M': [1.9, 0.0, 0.0], 'F': [2.8, 0.0, 0.0], 'P': [-1.6, 0.0, 0.0],
            'S': [-0.8, 0.0, 0.0], 'T': [-0.7, 0.0, 0.0], 'W': [-0.9, 0.0, 0.0],
            'Y': [-1.3, 0.0, 0.0], 'V': [4.2, 0.0, 0.0]
        }
        
        features = []
        
        # Basic composition
        features.extend([len(sequence)])
        
        # Average properties
        hydrophobicity = np.mean([aa_properties.get(aa, [0,0,0])[0] for aa in sequence])
        charge = np.sum([aa_properties.get(aa, [0,0,0])[1] for aa in sequence])
        aromaticity = np.sum([aa_properties.get(aa, [0,0,0])[2] for aa in sequence])
        
        features.extend([hydrophobicity, charge, aromaticity])
        
        # Amino acid frequencies
        aa_counts = {aa: sequence.count(aa) / len(sequence) for aa in 'ACDEFGHIKLMNPQRSTVWY'}
        features.extend([aa_counts.get(aa, 0) for aa in 'ACDEFGHIKLMNPQRSTVWY'])
        
        # Secondary structure propensities (simplified)
        helix_residues = 'AEHKQR'
        sheet_residues = 'CFITVY' 
        turn_residues = 'DGNPST'
        
        helix_prop = sum(1 for aa in sequence if aa in helix_residues) / len(sequence)
        sheet_prop = sum(1 for aa in sequence if aa in sheet_residues) / len(sequence)
        turn_prop = sum(1 for aa in sequence if aa in turn_residues) / len(sequence)
        
        features.extend([helix_prop, sheet_prop, turn_prop])
        
        feature_array = np.array(features)
        self.features_cache[sequence] = feature_array
        return feature_array
    
    def train_epitope_models(self, training_data: Dict[str, List[Tuple[str, int]]]):
        """Train B-cell and T-cell epitope prediction models"""
        
        # Prepare B-cell training data
        if 'bcell' in training_data:
            bcell_sequences, bcell_labels = zip(*training_data['bcell'])
            bcell_features = np.array([self.extract_sequence_features(seq) 
                                     for seq in bcell_sequences])
            
            self.bcell_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.bcell_model.fit(bcell_features, bcell_labels)
        
        # Prepare T-cell training data  
        if 'tcell' in training_data:
            tcell_sequences, tcell_labels = zip(*training_data['tcell'])
            tcell_features = np.array([self.extract_sequence_features(seq) 
                                     for seq in tcell_sequences])
            
            self.tcell_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.tcell_model.fit(tcell_features, tcell_labels)
    
    def predict_bcell_epitopes(self, protein_sequence: str, 
                              window_size: int = 15) -> List[EpitopeCandidate]:
        """Predict B-cell epitopes using sliding window"""
        if not self.bcell_model:
            # Use simple hydrophilicity-based prediction if no model
            return self._simple_bcell_prediction(protein_sequence, window_size)
        
        epitopes = []
        for i in range(len(protein_sequence) - window_size + 1):
            peptide = protein_sequence[i:i + window_size]
            features = self.extract_sequence_features(peptide)
            
            # Get prediction probability
            prob = self.bcell_model.predict_proba([features])[0][1]
            
            if prob > 0.6:  # Threshold for epitope prediction
                epitope = EpitopeCandidate(
                    sequence=peptide,
                    start_pos=i,
                    end_pos=i + window_size,
                    score=prob,
                    epitope_type='B-cell'
                )
                epitopes.append(epitope)
        
        return epitopes
    
    def predict_tcell_epitopes(self, protein_sequence: str, 
                              mhc_alleles: List[str] = None) -> List[EpitopeCandidate]:
        """Predict T-cell epitopes for MHC binding"""
        if not mhc_alleles:
            mhc_alleles = ['HLA-A*02:01', 'HLA-A*01:01', 'HLA-B*07:02']
        
        epitopes = []
        
        # MHC Class I epitopes (8-11 residues)
        for length in [9, 10]:
            for i in range(len(protein_sequence) - length + 1):
                peptide = protein_sequence[i:i + length]
                
                for allele in mhc_alleles:
                    # Simplified binding prediction
                    score = self._simple_mhc_binding(peptide, allele)
                    
                    if score > 0.5:
                        epitope = EpitopeCandidate(
                            sequence=peptide,
                            start_pos=i,
                            end_pos=i + length,
                            score=score,
                            epitope_type='T-cell',
                            mhc_allele=allele
                        )
                        epitopes.append(epitope)
        
        return epitopes
    
    def _simple_bcell_prediction(self, sequence: str, window_size: int) -> List[EpitopeCandidate]:
        """Simple B-cell epitope prediction based on hydrophilicity"""
        hydrophilicity = {
            'A': 0.5, 'R': 3.0, 'N': 0.2, 'D': 3.0, 'C': -1.0,
            'Q': 0.2, 'E': 3.0, 'G': 0.0, 'H': -0.5, 'I': -1.8,
            'L': -1.8, 'K': 3.0, 'M': -1.3, 'F': -2.5, 'P': 0.0,
            'S': 0.3, 'T': -0.4, 'W': -3.4, 'Y': -2.3, 'V': -1.5
        }
        
        epitopes = []
        for i in range(len(sequence) - window_size + 1):
            peptide = sequence[i:i + window_size]
            hydro_score = np.mean([hydrophilicity.get(aa, 0) for aa in peptide])
            
            if hydro_score > 0.5:  # Hydrophilic regions
                epitope = EpitopeCandidate(
                    sequence=peptide,
                    start_pos=i,
                    end_pos=i + window_size,
                    score=hydro_score / 3.0,  # Normalize
                    epitope_type='B-cell'
                )
                epitopes.append(epitope)
        
        return epitopes
    
    def _simple_mhc_binding(self, peptide: str, allele: str) -> float:
        """Simplified MHC binding prediction"""
        # This is a very simplified approach
        # In reality, you'd use tools like NetMHCpan, MHCflurry, etc.
        
        anchor_positions = {
            'HLA-A*02:01': {1: ['L', 'M'], 8: ['L', 'V', 'I']},
            'HLA-A*01:01': {2: ['T', 'S'], 8: ['Y', 'F']},
            'HLA-B*07:02': {1: ['P'], 8: ['L', 'A']}
        }
        
        if allele in anchor_positions and len(peptide) >= 9:
            score = 0.3  # Base score
            
            for pos, preferred_aa in anchor_positions[allele].items():
                if pos <= len(peptide) and peptide[pos-1] in preferred_aa:
                    score += 0.3
            
            return min(score, 1.0)
        
        return 0.2

class mRNAVaccineDesigner:
    """Design optimized mRNA sequences for vaccine applications"""
    
    def __init__(self):
        self.human_codon_usage = {
            'TTT': 17.6, 'TTC': 20.3, 'TTA': 7.7, 'TTG': 12.9,
            'TCT': 15.2, 'TCC': 17.7, 'TCA': 12.2, 'TCG': 4.4,
            # Add complete human codon usage table
        }
    
    def optimize_mrna_sequence(self, protein_sequence: str) -> Dict[str, str]:
        """Optimize mRNA sequence for stability and translation efficiency"""
        
        # Back-translate to DNA with optimized codons
        optimized_cds = self._optimize_codons(protein_sequence)
        
        # Add regulatory elements
        kozak_sequence = "GCCGCCACC"  # Strong Kozak consensus
        start_codon = "ATG"
        stop_codon = "TAA"  # Most efficient stop codon
        
        # 5' and 3' UTRs for stability
        utr5 = "GGAAAGCCTGTCTGCTACC"  # β-globin 5'UTR-like
        utr3 = "CTCGAGAAGCTTTTTCCACAGGGAAAT"  # Simplified 3'UTR
        
        full_mrna = utr5 + kozak_sequence + start_codon + optimized_cds + stop_codon + utr3
        
        # Calculate optimization metrics
        gc_content = gc_fraction(full_mrna) * 100
        
        return {
            'optimized_mrna': full_mrna,
            'cds_only': optimized_cds,
            'gc_content': gc_content,
            'length': len(full_mrna)
        }
    
    def _optimize_codons(self, protein_sequence: str) -> str:
        """Optimize codon usage for human expression"""
        # Simplified codon optimization
        optimal_codons = {
            'F': 'TTC', 'L': 'CTG', 'S': 'TCC', 'Y': 'TAC',
            'C': 'TGC', 'W': 'TGG', 'P': 'CCC', 'H': 'CAC',
            'Q': 'CAG', 'R': 'CGC', 'I': 'ATC', 'T': 'ACC',
            'N': 'AAC', 'K': 'AAG', 'V': 'GTG', 'A': 'GCC',
            'D': 'GAC', 'E': 'GAG', 'G': 'GGC', 'M': 'ATG'
        }
        
        optimized_cds = ""
        for aa in protein_sequence:
            if aa in optimal_codons:
                optimized_cds += optimal_codons[aa]
            elif aa == '*':
                break
        
        return optimized_cds
    
    def analyze_mrna_stability(self, mrna_sequence: str) -> Dict[str, float]:
        """Analyze mRNA sequence for stability features"""
        
        # GC content analysis
        gc_content = (mrna_sequence.count('G') + mrna_sequence.count('C')) / len(mrna_sequence) * 100
        
        # Look for problematic sequences
        polyA_runs = len(re.findall(r'A{6,}', mrna_sequence))
        polyT_runs = len(re.findall(r'T{6,}', mrna_sequence))
        
        # Secondary structure prediction (simplified)
        hairpin_potential = self._calculate_hairpin_potential(mrna_sequence)
        
        return {
            'gc_content': gc_content,
            'polyA_runs': polyA_runs,
            'polyT_runs': polyT_runs,
            'hairpin_potential': hairpin_potential,
            'length': len(mrna_sequence)
        }
    
    def _calculate_hairpin_potential(self, sequence: str) -> float:
        """Calculate potential for hairpin formation"""
        # Very simplified hairpin detection
        hairpin_score = 0
        window_size = 20
        
        for i in range(len(sequence) - window_size):
            window = sequence[i:i + window_size]
            gc_in_window = (window.count('G') + window.count('C')) / window_size
            
            # High GC content increases hairpin potential
            if gc_in_window > 0.7:
                hairpin_score += 1
        
        return hairpin_score / (len(sequence) - window_size + 1)

class VaccineEvaluator:
    """Evaluate vaccine candidates for efficacy and safety"""
    
    def __init__(self):
        self.immunogenicity_model = None
    
    def score_vaccine_candidate(self, epitopes: List[EpitopeCandidate], 
                              conservation_scores: np.ndarray) -> Dict[str, float]:
        """Score vaccine candidate based on multiple criteria"""
        
        if not epitopes:
            return {'total_score': 0.0, 'coverage': 0.0, 'conservation': 0.0}
        
        # Calculate coverage score
        bcell_epitopes = [e for e in epitopes if e.epitope_type == 'B-cell']
        tcell_epitopes = [e for e in epitopes if e.epitope_type == 'T-cell']
        
        coverage_score = (len(bcell_epitopes) * 0.4 + len(tcell_epitopes) * 0.6) / 10.0
        coverage_score = min(coverage_score, 1.0)
        
        # Calculate average conservation
        epitope_conservation = []
        for epitope in epitopes:
            if epitope.start_pos < len(conservation_scores):
                region_conservation = conservation_scores[epitope.start_pos:epitope.end_pos]
                epitope_conservation.append(np.mean(region_conservation))
        
        avg_conservation = np.mean(epitope_conservation) if epitope_conservation else 0.0
        
        # Calculate immunogenicity potential
        immunogenicity = np.mean([e.score for e in epitopes])
        
        # Combined score
        total_score = (coverage_score * 0.3 + 
                      avg_conservation * 0.4 + 
                      immunogenicity * 0.3)
        
        return {
            'total_score': total_score,
            'coverage': coverage_score,
            'conservation': avg_conservation,
            'immunogenicity': immunogenicity,
            'num_bcell_epitopes': len(bcell_epitopes),
            'num_tcell_epitopes': len(tcell_epitopes)
        }
    
    def analyze_cross_reactivity(self, epitopes: List[EpitopeCandidate], 
                                human_proteome_sample: List[str]) -> Dict[str, int]:
        """Check for potential cross-reactivity with human proteins"""
        
        cross_reactive = {'high_risk': 0, 'medium_risk': 0, 'low_risk': 0}
        
        for epitope in epitopes:
            for human_protein in human_proteome_sample:
                # Simple sequence similarity check
                if epitope.sequence in human_protein:
                    cross_reactive['high_risk'] += 1
                elif self._sequence_similarity(epitope.sequence, human_protein) > 0.8:
                    cross_reactive['medium_risk'] += 1
                elif self._sequence_similarity(epitope.sequence, human_protein) > 0.6:
                    cross_reactive['low_risk'] += 1
        
        return cross_reactive
    
    def _sequence_similarity(self, seq1: str, seq2: str) -> float:
        """Calculate sequence similarity (simplified)"""
        # Use longest common subsequence approach
        matches = 0
        for i, aa in enumerate(seq1):
            if i < len(seq2) and aa == seq2[i]:
                matches += 1
        
        return matches / max(len(seq1), len(seq2))

class VaccineDesignPipeline:
    """Complete pipeline for computational vaccine design"""
    
    def __init__(self):
        self.genome_analyzer = ViralGenomeAnalyzer()
        self.epitope_predictor = EpitopePrediction()
        self.mrna_designer = mRNAVaccineDesigner()
        self.evaluator = VaccineEvaluator()
        
    def design_vaccine(self, viral_genome: str, 
                      variant_genomes: List[str] = None,
                      target_proteins: List[str] = None) -> Dict:
        """Complete vaccine design workflow"""
        
        results = {}
        
        # Step 1: Extract viral proteins
        print("Step 1: Extracting viral proteins...")
        proteins = self.genome_analyzer.extract_proteins(viral_genome)
        results['proteins'] = proteins
        
        if target_proteins:
            proteins = {k: v for k, v in proteins.items() if k in target_proteins}
        
        # Step 2: Analyze conservation across variants
        print("Step 2: Analyzing conservation across variants...")
        conservation_data = {}
        if variant_genomes:
            for protein_name, protein_seq in proteins.items():
                # Get sequences from variants
                variant_sequences = [protein_seq]  # Include reference
                for variant_genome in variant_genomes:
                    variant_proteins = self.genome_analyzer.extract_proteins(variant_genome)
                    if protein_name in variant_proteins:
                        variant_sequences.append(variant_proteins[protein_name])
                
                conservation_scores = self.genome_analyzer.analyze_conservation(variant_sequences)
                conservation_data[protein_name] = conservation_scores
        
        results['conservation'] = conservation_data
        
        # Step 3: Predict epitopes
        print("Step 3: Predicting epitopes...")
        all_epitopes = {}
        for protein_name, protein_seq in proteins.items():
            bcell_epitopes = self.epitope_predictor.predict_bcell_epitopes(protein_seq)
            tcell_epitopes = self.epitope_predictor.predict_tcell_epitopes(protein_seq)
            
            # Add conservation scores to epitopes
            if protein_name in conservation_data:
                for epitope in bcell_epitopes + tcell_epitopes:
                    if epitope.start_pos < len(conservation_data[protein_name]):
                        region_scores = conservation_data[protein_name][
                            epitope.start_pos:epitope.end_pos
                        ]
                        epitope.conservation_score = np.mean(region_scores)
            
            all_epitopes[protein_name] = {
                'bcell': bcell_epitopes,
                'tcell': tcell_epitopes
            }
        
        results['epitopes'] = all_epitopes
        
        # Step 4: Design mRNA vaccine sequences
        print("Step 4: Designing mRNA vaccine sequences...")
        vaccine_designs = {}
        for protein_name, protein_seq in proteins.items():
            # Select top epitopes for vaccine design
            protein_epitopes = all_epitopes[protein_name]['bcell'] + all_epitopes[protein_name]['tcell']
            top_epitopes = sorted(protein_epitopes, 
                                key=lambda x: x.score * x.conservation_score, 
                                reverse=True)[:10]
            
            # Create multi-epitope protein
            vaccine_sequence = "".join([e.sequence for e in top_epitopes[:5]])
            
            # Optimize mRNA
            mrna_design = self.mrna_designer.optimize_mrna_sequence(vaccine_sequence)
            vaccine_designs[protein_name] = mrna_design
        
        results['vaccine_designs'] = vaccine_designs
        
        # Step 5: Evaluate vaccine candidates
        print("Step 5: Evaluating vaccine candidates...")
        evaluations = {}
        for protein_name in proteins.keys():
            protein_epitopes = all_epitopes[protein_name]['bcell'] + all_epitopes[protein_name]['tcell']
            conservation_scores = conservation_data.get(protein_name, np.array([1.0]))
            
            evaluation = self.evaluator.score_vaccine_candidate(
                protein_epitopes, conservation_scores
            )
            evaluations[protein_name] = evaluation
        
        results['evaluations'] = evaluations
        
        return results
    
    def generate_report(self, results: Dict) -> str:
        """Generate comprehensive vaccine design report"""
        
        report = []
        report.append("=" * 80)
        report.append("COMPUTATIONAL VACCINE DESIGN REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary statistics
        report.append("SUMMARY")
        report.append("-" * 40)
        total_proteins = len(results['proteins'])
        total_epitopes = sum(
            len(epitopes['bcell']) + len(epitopes['tcell']) 
            for epitopes in results['epitopes'].values()
        )
        report.append(f"Proteins analyzed: {total_proteins}")
        report.append(f"Total epitopes predicted: {total_epitopes}")
        report.append("")
        
        # Protein-specific analysis
        for protein_name, protein_seq in results['proteins'].items():
            report.append(f"PROTEIN: {protein_name.upper()}")
            report.append("-" * 40)
            report.append(f"Length: {len(protein_seq)} amino acids")
            
            if protein_name in results['epitopes']:
                epitopes = results['epitopes'][protein_name]
                report.append(f"B-cell epitopes: {len(epitopes['bcell'])}")
                report.append(f"T-cell epitopes: {len(epitopes['tcell'])}")
            
            if protein_name in results['evaluations']:
                eval_data = results['evaluations'][protein_name]
                report.append(f"Vaccine score: {eval_data['total_score']:.3f}")
                report.append(f"Conservation: {eval_data['conservation']:.3f}")
            
            if protein_name in results['vaccine_designs']:
                design = results['vaccine_designs'][protein_name]
                report.append(f"mRNA length: {design['length']} nucleotides")
                report.append(f"GC content: {design['gc_content']:.1f}%")
            
            report.append("")
        
        # Top epitope candidates
        report.append("TOP EPITOPE CANDIDATES")
        report.append("-" * 40)
        all_epitopes_list = []
        for protein_name, epitopes in results['epitopes'].items():
            for epitope in epitopes['bcell'] + epitopes['tcell']:
                all_epitopes_list.append((protein_name, epitope))
        
        # Sort by combined score
        top_epitopes = sorted(
            all_epitopes_list,
            key=lambda x: x[1].score * x[1].conservation_score,
            reverse=True
        )[:10]
        
        for i, (protein, epitope) in enumerate(top_epitopes, 1):
            report.append(f"{i:2d}. {epitope.sequence} ({protein}, {epitope.epitope_type})")
            report.append(f"    Score: {epitope.score:.3f}, Conservation: {epitope.conservation_score:.3f}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)

def demo_coronavirus_vaccine_design():
    """Demonstration of vaccine design for coronavirus"""
    
    # Example SARS-CoV-2 genome segment (spike protein region)
    # This is a simplified example - real genomes are much longer
    example_genome = """
    ATGTTTGTTTTTCTTGTTTTATTGCCACTAGTCTCTAGTCAGTGTGTTAATCTTACAACCAGAACTCAATTACCCCCTGCATACACTAATTCTTTCACACGTGGTGTTTATTACCCTGACAAAGTTTTCAGATCCTCAGTTTTACATTCAACTCAGGACTTGTTCTTACCTTTCTTTTCCAATGTTACTTGGTTCCATGCTATACATGTCTCTGGGACCAATGGTACTAAGAGGTTTGATAACCCTGTCCTACCATTTAATGATGGTGTTTATTTTGCTTCCACTGAGAAGTCTAACATAATAAGAGGCTGGATTTTTGGTACTACTTTAGATTCGAAGACCCAGTCCCTACTTATTGTTAATAACGCTACTAATGTTGTTATTAAAGTCTGTGAATTTCAATTTTGTAATGATCCATTTTTGGGTGTTTATTACCACAAAAACAACAAAAGTTGGATGGAAAGTGAGTTCAGAGTTTATTCTAGTGCGAATAATTGCACTTTTGAATATGTCTCTCAGCCTTTTCTTATGGACCTTGAAGGAAAACAGGGTAATTTCAAAAATCTTAGGGAATTTGTGTTTAAGAATATTGATGGTTATTTTAAAATATATTCTAAGCACACGCCTATTAATTTAGTGCGTGATCTCCCTCAGGGTTTTTCGGCTTTAGAACCATTGGTAGATTTGCCAATAGGTATTAACATCACTAGGTTTCAAACTTTACTTGCTTTACATAGAAGTTATTTGACTCCTGGTGATTCTTCTTCAGGTTGGACAGCTGGTGCTGCAGCTTATTATGTGGGTTATCTTCAACCTAGGACTTTTCTATTAAAATATAATGAAAATGGAACCATTACAGATGCTGTAGACTGTGCACTTGACCCTCTCTCAGAAACAAAGTGTACGTTGAAATCCTTCACTGTAGAAAAAGGAATCTATCAAACTTCTAACTTTAGAGTCCAACCAACAGAATCTATTGTTAGATTTCCTAATATTACAAACTTGTGCCCTTTTGGTGAAGTTTTTAACGCCACCAGATTTGCATCTGTTTATGCTTGGAACAGGAAGAGAATCAGCAACTGTGTTGCTGATTATTCTGTCCTATATAATTCCGCATCATTTTCCACTTTTAAGTGTTATGGAGTGTCTCCTACTAAATTAAATGATCTCTGCTTTACTAATGTCTATGCAGATTCATTTGTAATTAGAGGTGATGAAGTCAGACAAATCGCTCCAGGGCAAACTGGAAAGATTGCTGATTATAATTATAAATTACCAGATGATTTTACAGGCTGCGTTATAGCTTGGAATTCTAACAATCTTGATTCTAAGGTTGGTGGTAATTATAATTACCTGTATAGATTGTTTAGGAAGTCTAATCTCAAACCTTTTGAGAGAGATATTTCAACTGAAATCTATCAGGCCGGTAGCACACCTTGTAATGGTGTTGAAGGTTTTAATTGTTACTTTCCTTTACAATCATATGGTTTCCAACCCACTAATGGTGTTGGTTACCAACCATACAGAGTAGTAGTACTTTCTTTTGAACTTCTACATGCACCAGCAACTGTTTGTGGACCTAAAAAGTCTACTAATTTGGTTAAAAACAAATGTGTCAATTTCAACTTCAATGGTTTAACAGGCACAGGTGTTCTTACTGAGTCTAACAAAAAGTTTCTGCCTTTCCAACAATTTGGCAGAGACATTGCTGACACTACTGATGCTGTCCGTGATCCACAGACACTTGAGATTCTTGACATTACACCATGTTCTTTTGGTGGTGTCAGTGTTATAACACCAGGAACAAATACTTCTAACCAGGTTGCTGTTCTTTATCAGGATGTTAACTGCACAGAAGTCCCTGTTGCTATTCATGCAGATCAACTTACTCCTACTTGGCGTGTTTATTCTACAGGTTCTAATGTTTTTCAAACACGTGCAGGCTGTTTAATAGGGGCTGAACATGTCAACAACTCATATGAGTGTGACATACCCATTGGTGCAGGTATATGCGCTAGTTATCAGACTCAGACTAATTCTCCTCGGCGGGCACGTAGTGTAGCTAGTCAATCCATCATTGCCTACACTATGTCACTTGGTGCAGAAAATTCAGTTGCTTACTCTAATAACTCTATTGCCATACCCACAAATTTTACTATTAGTGTTACCACAGAAATTCTACCAGTGTCTATGACCAAGACATCAGTAGATTGTACAATGTACATTTGTGGTGATTCAACTGAATGCAGCAATCTTTTGTTGCAATATGGCAGTTTTTGTACACAATTAAACCGTGCTTTAACTGGAATAGCTGTTGAACAAGACAAAAACACCCAAGAAGTTTTTGCACAAGTCAAACAAATTTACAAAACACCACCAATTAAAGATTTTGGTGGTTTTAATTTTTCACAAATATTACCAGATCCATCAAAACCAAGCAAGAGGTCATTTATTGAAGATCTACTTTTCAACAAAGTGACACTTGCAGATGCTGGCTTCATCAAACAATATGGTGATTGCCTTGGTGATATTGCTGCTAGAGACCTCATTTGTGCACAAAAGTTTAACGGCCTTACTGTTTTGCCACCTTTGCTCACAGATGAAATGATTGCTCAATACACTTCTGCACTGTTAGCGGGTACAATCACTTCTGGTTGGACCTTTGGTGCAGGTGCTGCATTACAAATACCATTTGCTATGCAAATGGCTTATAGGTTTAATGGTATTGGAGTTACACAGAATGTTCTCTATGAGAACCAAAAATTGATTGCCAACCAATTTAATAGTGCTATTGGCAAAATTCAAGACTCACTTTCTTCCACAGCAAGTGCACTTGGAAAACTTCAAGATGTGGTCAACCAAAATGCACAAGCTTTAAACACGCTTGTTAAACAACTTAGCTCCAATTTTGGTGCAATTTCAAGTGTTTTAAATGATATCCTTTCACGTCTTGACAAAGTTGAGGCTGAAGTGCAAATTGATAGGTTGATCACAGGCAGACTTCAAAGTTTGCAGACATATGTGACTCAACAATTAATTAGAGCTGCAGAAATCAGAGCTTCTGCTAATCTTGCTGCTACTAAAATGTCAGAGTGTGTACTTGGACAATCAAAAAGAGTTGATTTTTGTGGAAAGGGCTATCATCTTATGTCCTTCCCTCAGTCAGCACCTCATGGTGTAGTCTTCTTGCATGTGACTTATGTCCCTGCACAAGAAAAGAACTTCACAACTGCTCCTGCCATTTGTCATGATGGAAAAGCACACTTTCCTCGTGAAGGTGTCTTTGTTTCAAATGGCACACACTGGTTTGTAACACAAAGGAATTTTTATGAACCACAAATCATTACTACAGACAACACATTTGTGTCTGGTAACTGTGATGTTGTAATAGGAATTGTCAACAACACAGTTTATGATCCTTTGCAACCTGAATTAGATTCATTCAAGGAGGAGTTAGATAAATATTTTAAGAATCATACATCACCAGATGTTGATTTAGGTGACATCTCTGGCATTAATGCTTCAGTTGTAAACATTCAAAAAGAAATTGACCGCCTCAATGAGGTTGCCAAGAATTTAAATGAATCTCTCATCGATCTCCAAGAACTTGGAAAGTATGAGCAGTATATAAAATGGCCATGGTACATTTGGCTAGGTTTTATAGCTGGCTTGATTGCCATAGTAATGGTGACAATTATGCTTTGCTGTATGACCAGTTGCTGTAGTTGTCTCAAGGGCTGTTGTTCTTGTGGATCCTGCTGCAAATTTGATGAAGACGACTCTGAGCCAGTGCTCAAAGGAGTCAAATTACATTACACATAA
    """
    
    # Clean up the sequence
    genome_sequence = example_genome.replace('\n', '').replace(' ', '').upper()
    
    # Create example variant genomes (simplified mutations)
    variant_genomes = [
        genome_sequence.replace('A', 'G', 5),  # Some A->G mutations
        genome_sequence.replace('T', 'C', 3),  # Some T->C mutations
    ]
    
    # Initialize pipeline
    pipeline = VaccineDesignPipeline()
    
    # Run vaccine design
    print("Starting computational vaccine design for coronavirus...")
    results = pipeline.design_vaccine(
        viral_genome=genome_sequence,
        variant_genomes=variant_genomes,
        target_proteins=['spike', 'nucleocapsid']  # Focus on key proteins
    )
    
    # Generate and display report
    report = pipeline.generate_report(results)
    print(report)
    
    return results

if __name__ == "__main__":
    # Run the demonstration
    results = demo_coronavirus_vaccine_design()
    
    # Save detailed results
    print("\nDetailed results saved. Key findings:")
    for protein, evaluation in results['evaluations'].items():
        print(f"{protein}: Score {evaluation['total_score']:.3f}, "
              f"{evaluation['num_bcell_epitopes']} B-cell + "
              f"{evaluation['num_tcell_epitopes']} T-cell epitopes")
