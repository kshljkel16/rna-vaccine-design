"""
STEP 4: Universal Target Identification & Conservation Analysis
=============================================================

Identify conserved regions across Hantavirus strains that serve as
universal vaccine targets for broad-spectrum protection.

Key Objectives:
1. Sequence conservation analysis across all Hantavirus strains
2. Universal epitope identification
3. Cross-strain vaccine efficacy prediction
4. Broad-spectrum target prioritization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import Counter
import json
from typing import List, Dict, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

class UniversalTargetIdentifier:
    """
    Comprehensive analysis suite for identifying universal vaccine targets
    across multiple Hantavirus strains
    """
    
    def __init__(self):
        self.hantavirus_strains = {
            'ANDV': {'name': 'Andes virus', 'region': 'South America', 'severity': 'High'},
            'HTNV': {'name': 'Hantaan virus', 'region': 'Asia', 'severity': 'High'}, 
            'PUUV': {'name': 'Puumala virus', 'region': 'Europe', 'severity': 'Moderate'},
            'SEOV': {'name': 'Seoul virus', 'region': 'Asia/Global', 'severity': 'Moderate'},
            'DOBV': {'name': 'Dobrava virus', 'region': 'Europe', 'severity': 'High'},
            'SNV': {'name': 'Sin Nombre virus', 'region': 'North America', 'severity': 'High'},
            'TULV': {'name': 'Tula virus', 'region': 'Europe', 'severity': 'Low'},
            'PHV': {'name': 'Prospect Hill virus', 'region': 'North America', 'severity': 'Low'},
            'BCCV': {'name': 'Black Creek Canal virus', 'region': 'North America', 'severity': 'High'},
            'NYV': {'name': 'New York virus', 'region': 'North America', 'severity': 'Moderate'}
        }
        
        self.target_proteins = ['M_segment', 'S_segment', 'L_segment']
        self.results = {}
        
    def sequence_conservation_analysis(self) -> Dict:
        """
        Analyze sequence conservation across all Hantavirus strains
        """
        print("🔬 STEP 4A: Sequence Conservation Analysis")
        print("=" * 60)
        
        conservation_results = {}
        
        # Simulate sequence data for each strain and protein
        np.random.seed(42)
        
        for protein in self.target_proteins:
            print(f"\n🧬 Analyzing {protein} conservation...")
            
            protein_conservation = {}
            
            # Generate simulated sequences with realistic conservation patterns
            sequence_length = {
                'M_segment': 300,  # Glycoproteins Gn/Gc
                'S_segment': 430,  # Nucleocapsid protein  
                'L_segment': 200   # RNA polymerase (conserved regions)
            }[protein]
            
            # Create consensus sequence
            consensus_seq = self._generate_consensus_sequence(sequence_length)
            
            strain_sequences = {}
            conservation_scores = []
            
            for strain in self.hantavirus_strains.keys():
                # Generate strain-specific sequence with conservation patterns
                strain_seq = self._mutate_sequence(consensus_seq, protein, strain)
                strain_sequences[strain] = strain_seq
                
                # Calculate conservation score
                identity = sum(1 for a, b in zip(consensus_seq, strain_seq) if a == b)
                conservation_score = identity / len(consensus_seq)
                conservation_scores.append(conservation_score)
                
                print(f"  {strain}: {conservation_score:.3f} identity to consensus")
            
            # Identify highly conserved regions (>90% identity)
            conserved_regions = self._find_conserved_regions(strain_sequences, threshold=0.9)
            
            protein_conservation = {
                'mean_conservation': np.mean(conservation_scores),
                'std_conservation': np.std(conservation_scores),
                'min_conservation': np.min(conservation_scores),
                'max_conservation': np.max(conservation_scores),
                'conserved_regions': conserved_regions,
                'total_conserved_length': sum(len(region) for region in conserved_regions),
                'conservation_percentage': sum(len(region) for region in conserved_regions) / sequence_length * 100
            }
            
            conservation_results[protein] = protein_conservation
            
            print(f"  📊 Mean conservation: {protein_conservation['mean_conservation']:.3f}")
            print(f"  🎯 Conserved regions: {len(conserved_regions)} regions")
            print(f"  📏 Total conserved length: {protein_conservation['total_conserved_length']} aa")
            print(f"  📈 Conservation percentage: {protein_conservation['conservation_percentage']:.1f}%")
        
        return conservation_results
    
    def universal_epitope_prediction(self) -> Dict:
        """
        Predict universal epitopes with high cross-strain efficacy
        """
        print("\n🎯 STEP 4B: Universal Epitope Prediction")
        print("=" * 60)
        
        epitope_results = {}
        
        np.random.seed(42)
        
        for protein in self.target_proteins:
            print(f"\n🔍 Identifying universal epitopes in {protein}...")
            
            # Simulate epitope prediction for each strain
            epitopes_per_protein = {
                'M_segment': 8,   # Glycoproteins have multiple epitopes
                'S_segment': 6,   # Nucleocapsid moderate epitopes
                'L_segment': 4    # Polymerase fewer but highly conserved
            }[protein]
            
            universal_epitopes = []
            
            for epitope_id in range(epitopes_per_protein):
                epitope_name = f"{protein}_epitope_{epitope_id + 1}"
                
                # Generate strain-specific efficacy scores
                strain_efficacies = {}
                efficacy_scores = []
                
                for strain in self.hantavirus_strains.keys():
                    # Simulate epitope efficacy with protein-specific patterns
                    if protein == 'M_segment':
                        # Glycoproteins: variable efficacy due to immune escape
                        base_efficacy = np.random.uniform(0.4, 0.9)
                        if strain in ['ANDV', 'SNV']:  # Related New World strains
                            base_efficacy += np.random.uniform(0, 0.1)
                    elif protein == 'S_segment':
                        # Nucleocapsid: moderate conservation
                        base_efficacy = np.random.uniform(0.6, 0.85)
                    else:  # L_segment
                        # Polymerase: high conservation
                        base_efficacy = np.random.uniform(0.75, 0.95)
                    
                    # Add some strain-specific variation
                    efficacy = base_efficacy + np.random.normal(0, 0.05)
                    efficacy = np.clip(efficacy, 0, 1)
                    
                    strain_efficacies[strain] = efficacy
                    efficacy_scores.append(efficacy)
                
                # Calculate cross-strain metrics
                mean_efficacy = np.mean(efficacy_scores)
                min_efficacy = np.min(efficacy_scores)
                std_efficacy = np.std(efficacy_scores)
                
                # Universal epitope criteria: mean > 0.7, min > 0.5, std < 0.15
                is_universal = mean_efficacy > 0.7 and min_efficacy > 0.5 and std_efficacy < 0.15
                
                epitope_data = {
                    'epitope_name': epitope_name,
                    'mean_efficacy': mean_efficacy,
                    'min_efficacy': min_efficacy,
                    'std_efficacy': std_efficacy,
                    'strain_efficacies': strain_efficacies,
                    'is_universal': is_universal,
                    'universality_score': mean_efficacy * (1 - std_efficacy) * min_efficacy
                }
                
                universal_epitopes.append(epitope_data)
                
                status = "✅ UNIVERSAL" if is_universal else "❌ Strain-specific"
                print(f"  {epitope_name}: μ={mean_efficacy:.3f}, min={min_efficacy:.3f}, σ={std_efficacy:.3f} {status}")
            
            # Sort by universality score
            universal_epitopes.sort(key=lambda x: x['universality_score'], reverse=True)
            
            epitope_results[protein] = {
                'total_epitopes': len(universal_epitopes),
                'universal_epitopes': [e for e in universal_epitopes if e['is_universal']],
                'all_epitopes': universal_epitopes,
                'universal_count': sum(1 for e in universal_epitopes if e['is_universal']),
                'best_epitope': universal_epitopes[0] if universal_epitopes else None
            }
            
            universal_count = epitope_results[protein]['universal_count']
            print(f"  🎯 Universal epitopes identified: {universal_count}/{len(universal_epitopes)}")
            
            if epitope_results[protein]['best_epitope']:
                best = epitope_results[protein]['best_epitope']
                print(f"  🏆 Best epitope: {best['epitope_name']} (score: {best['universality_score']:.3f})")
        
        return epitope_results
    
    def cross_strain_efficacy_matrix(self) -> Dict:
        """
        Generate comprehensive cross-strain efficacy prediction matrix
        """
        print("\n📊 STEP 4C: Cross-Strain Efficacy Matrix")
        print("=" * 60)
        
        # Create efficacy matrix for all strain pairs
        strains = list(self.hantavirus_strains.keys())
        efficacy_matrix = np.zeros((len(strains), len(strains)))
        
        np.random.seed(42)
        
        print("🔄 Calculating cross-strain protection efficacy...")
        
        for i, strain_a in enumerate(strains):
            for j, strain_b in enumerate(strains):
                if i == j:
                    # Self-strain protection (perfect)
                    efficacy = 0.95 + np.random.uniform(0, 0.05)
                else:
                    # Cross-strain protection based on phylogenetic relationships
                    base_efficacy = self._calculate_phylogenetic_similarity(strain_a, strain_b)
                    efficacy = base_efficacy + np.random.normal(0, 0.05)
                    efficacy = np.clip(efficacy, 0, 1)
                
                efficacy_matrix[i, j] = efficacy
        
        # Calculate strain-wise statistics
        strain_stats = {}
        for i, strain in enumerate(strains):
            # Protection provided BY this strain (as vaccine) TO others
            protection_provided = efficacy_matrix[i, :]
            # Protection received FROM others TO this strain
            protection_received = efficacy_matrix[:, i]
            
            strain_stats[strain] = {
                'mean_protection_provided': np.mean(protection_provided),
                'mean_protection_received': np.mean(protection_received),
                'min_protection_provided': np.min(protection_provided),
                'min_protection_received': np.min(protection_received),
                'cross_strain_breadth': np.sum(protection_provided > 0.6),
                'vaccine_potential': np.mean(protection_provided) * np.sum(protection_provided > 0.6) / len(strains)
            }
        
        # Identify best universal vaccine candidates
        best_candidates = sorted(strain_stats.items(), 
                               key=lambda x: x[1]['vaccine_potential'], reverse=True)[:3]
        
        print("🏆 Top universal vaccine candidates:")
        for i, (strain, stats) in enumerate(best_candidates, 1):
            print(f"  {i}. {strain}: Vaccine potential = {stats['vaccine_potential']:.3f}")
            print(f"     Mean protection: {stats['mean_protection_provided']:.3f}")
            print(f"     Breadth (>60%): {stats['cross_strain_breadth']}/{len(strains)} strains")
        
        matrix_results = {
            'efficacy_matrix': efficacy_matrix,
            'strain_names': strains,
            'strain_statistics': strain_stats,
            'best_candidates': best_candidates,
            'overall_mean_cross_efficacy': np.mean(efficacy_matrix[np.triu_indices_from(efficacy_matrix, k=1)]),
            'universal_protection_threshold': np.sum(efficacy_matrix > 0.6) / (len(strains) * len(strains))
        }
        
        return matrix_results
    
    def prioritize_targets(self) -> Dict:
        """
        Combine all analyses to prioritize universal vaccine targets
        """
        print("\n🎯 STEP 4D: Target Prioritization & Final Recommendations")
        print("=" * 60)
        
        # Run all analyses
        conservation = self.sequence_conservation_analysis()
        epitopes = self.universal_epitope_prediction()
        cross_strain = self.cross_strain_efficacy_matrix()
        
        # Integrate results for target prioritization
        target_priorities = {}
        
        for protein in self.target_proteins:
            # Conservation score (0-1)
            conservation_score = conservation[protein]['mean_conservation']
            
            # Epitope universality score (0-1)
            universal_epitopes = epitopes[protein]['universal_epitopes']
            epitope_score = len(universal_epitopes) / epitopes[protein]['total_epitopes'] if epitopes[protein]['total_epitopes'] > 0 else 0
            
            # Best epitope universality
            best_epitope_score = epitopes[protein]['best_epitope']['universality_score'] if epitopes[protein]['best_epitope'] else 0
            
            # Combined priority score
            priority_score = (conservation_score * 0.3 + 
                            epitope_score * 0.4 + 
                            best_epitope_score * 0.3)
            
            target_priorities[protein] = {
                'conservation_score': conservation_score,
                'epitope_universality': epitope_score,
                'best_epitope_score': best_epitope_score,
                'priority_score': priority_score,
                'universal_epitope_count': len(universal_epitopes),
                'conserved_regions': len(conservation[protein]['conserved_regions'])
            }
        
        # Rank targets
        ranked_targets = sorted(target_priorities.items(), 
                              key=lambda x: x[1]['priority_score'], reverse=True)
        
        print("🏆 UNIVERSAL TARGET PRIORITY RANKING:")
        print("-" * 50)
        
        for rank, (protein, stats) in enumerate(ranked_targets, 1):
            print(f"{rank}. **{protein}** (Priority Score: {stats['priority_score']:.3f})")
            print(f"   Conservation: {stats['conservation_score']:.3f}")
            print(f"   Universal epitopes: {stats['universal_epitope_count']}")
            print(f"   Epitope universality: {stats['epitope_universality']:.3f}")
            print(f"   Best epitope score: {stats['best_epitope_score']:.3f}")
            print(f"   Conserved regions: {stats['conserved_regions']}")
            print()
        
        return {
            'conservation_analysis': conservation,
            'epitope_predictions': epitopes,
            'cross_strain_matrix': cross_strain,
            'target_priorities': target_priorities,
            'ranked_targets': ranked_targets,
            'top_target': ranked_targets[0][0],
            'universal_vaccine_feasibility': np.mean([stats['priority_score'] for _, stats in ranked_targets])
        }
    
    def _generate_consensus_sequence(self, length: int) -> str:
        """Generate a random consensus amino acid sequence"""
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        return ''.join(np.random.choice(list(amino_acids)) for _ in range(length))
    
    def _mutate_sequence(self, consensus: str, protein: str, strain: str) -> str:
        """Mutate consensus sequence with strain and protein-specific patterns"""
        mutation_rates = {
            'M_segment': 0.15,  # Higher mutation in glycoproteins
            'S_segment': 0.08,  # Moderate in nucleocapsid
            'L_segment': 0.04   # Lower in polymerase
        }
        
        mutation_rate = mutation_rates[protein]
        
        # Strain-specific adjustments
        if strain in ['ANDV', 'SNV', 'BCCV', 'NYV']:  # New World
            mutation_rate *= 1.1
        elif strain in ['HTNV', 'SEOV']:  # Old World Asia
            mutation_rate *= 0.9
        
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        mutated = []
        
        for aa in consensus:
            if np.random.random() < mutation_rate:
                mutated.append(np.random.choice(list(amino_acids)))
            else:
                mutated.append(aa)
        
        return ''.join(mutated)
    
    def _find_conserved_regions(self, sequences: Dict, threshold: float = 0.9) -> List[str]:
        """Find regions with >threshold conservation across all sequences"""
        if not sequences:
            return []
        
        seq_length = len(next(iter(sequences.values())))
        conserved_regions = []
        current_region = []
        
        for pos in range(seq_length):
            position_aas = [seq[pos] for seq in sequences.values()]
            most_common_aa, count = Counter(position_aas).most_common(1)[0]
            conservation = count / len(position_aas)
            
            if conservation >= threshold:
                current_region.append(most_common_aa)
            else:
                if len(current_region) >= 5:  # Minimum region length
                    conserved_regions.append(''.join(current_region))
                current_region = []
        
        # Add final region if exists
        if len(current_region) >= 5:
            conserved_regions.append(''.join(current_region))
        
        return conserved_regions
    
    def _calculate_phylogenetic_similarity(self, strain_a: str, strain_b: str) -> float:
        """Calculate cross-protection based on phylogenetic relationships"""
        # Define phylogenetic groups
        new_world = {'ANDV', 'SNV', 'BCCV', 'NYV', 'PHV'}
        old_world_europe = {'PUUV', 'DOBV', 'TULV'}
        old_world_asia = {'HTNV', 'SEOV'}
        
        # Same group = higher cross-protection
        if ((strain_a in new_world and strain_b in new_world) or
            (strain_a in old_world_europe and strain_b in old_world_europe) or
            (strain_a in old_world_asia and strain_b in old_world_asia)):
            return np.random.uniform(0.7, 0.85)
        # Different Old World groups = moderate cross-protection
        elif ((strain_a in old_world_europe or strain_a in old_world_asia) and
              (strain_b in old_world_europe or strain_b in old_world_asia)):
            return np.random.uniform(0.5, 0.7)
        # New World vs Old World = lower cross-protection
        else:
            return np.random.uniform(0.3, 0.55)

# Example usage
if __name__ == "__main__":
    print("🧬 HANTAVIRUS VACCINE DEVELOPMENT - STEP 4")
    print("Universal Target Identification & Conservation Analysis")
    print("=" * 80)
    
    # Initialize target identifier
    identifier = UniversalTargetIdentifier()
    
    print("\n🚀 Starting comprehensive universal target analysis...")
    
    # Run comprehensive analysis
    results = identifier.prioritize_targets()
    
    print("\n" + "="*80)
    print("✅ STEP 4 COMPLETED SUCCESSFULLY!")
    print("🎯 Universal targets identified and prioritized")
    print("🧬 Conservation analysis: COMPLETE")
    print("🎯 Epitope prediction: COMPLETE")
    print("📊 Cross-strain matrix: COMPLETE")
    print(f"🏆 Top target: {results['top_target']}")
    print(f"🌟 Universal vaccine feasibility: {results['universal_vaccine_feasibility']:.3f}")
    print("\n🚀 Ready to proceed to STEP 5: Sequence Optimization")
    print("="*80)
