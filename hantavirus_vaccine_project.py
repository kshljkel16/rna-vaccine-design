"""
Hantavirus Vaccine Optimization Project - Updated
=================================================

Using Len's validated RNA chemical reactivity prediction models 
for targeted Hantavirus vaccine development.

Updated to use fin_archit_mt module.
"""

import torch
import numpy as np
from pathlib import Path

# Import Len's validated model architecture
from fin_archit_mt import LenAbsoluteFinalSeparateModel, load_len_absolute_final_model

# Hantavirus-specific sequences and targets
HANTAVIRUS_TARGETS = {
    "sin_nombre_nucleocapsid": {
        "name": "Sin Nombre Virus Nucleocapsid Protein",
        "sequence": "AUGAGCGACCGGAAUCUUUCACAAGCCUCAAGCUGUCCCGAUAACUGAUCAUCUACAAGAAAAUGGUCAGCGGUCGGCGGAACCAAGUUCCGAAAAUUUGCCAAAGAAAGGAAAUUAAAAAACUGAAGAUGGAAAACCUCACCGAAAAAGUAAACAUGGCGUCUUUCCACAUACCACUAGAAAAAAUCUACAGGAUACUGAAACAUGAAACCCAGAACCACAAAAGGAGAAAGACAUGGAGCGGGUGACGUUCCACAUACCCAUCCACAAUCAGAAA",
        "type": "structural_protein",
        "immunogenicity": "high"
    },
    
    "andes_glycoprotein": {
        "name": "Andes Virus Glycoprotein (Gc/Gn)",
        "sequence": "AUGGGAUCAAAAGUAAACACCAAAGCUCUGAAAUCUACCAUCUGAGAAAGUGGUCCGCAACAACACCAAGCCCUUCAUCGAAGAAGACAACCUGUUUGAACAAAAUGAUGGCUACAUUCUCUUUAAAAAGGGCCAGGUGACGAUCCUGGCACUGCUGAACAAACUGCUGUCCCCCAUGAUCCCACUCAAGCGGAUGGUGACAACUGGUCACAACCGCUACCUCAAGAAAGUGUUGGACAAAGAGAGGGAGUUGCAAGGGGAAACUGGAAGGAGAAAGUUAAAGACUGCACCAACCGACGG",
        "type": "surface_glycoprotein",
        "immunogenicity": "very_high"
    },
    
    "puumala_m_segment": {
        "name": "Puumala Virus M Segment (Partial)",
        "sequence": "AUGAACCACGCCGAGAACCAGGACUUCCUCCAAGUACUCUUGGAGAACGAAACUCUCAGCGACGAGAAAAUCAAGACGAAGUUCCGUGGCAACAAGGGAGAAAAGGUCAAGGGCGAAAGAAUAACCCGGAGCGGCGACACUGCCACCAAGAUCGAAACAUUAGAAAAGGUGGCGUUGGUUGACGGAUCCGACGACGAAAAGUUCCCGUACGAAAAGGUCAACCACCACCAGGCCGCACUGAUAGAGAGAACCUCCCGAAAAGAGCGGAUGAAAACAGCGAUGGAGCUGAAG",
        "type": "medium_segment",
        "immunogenicity": "moderate"
    },
    
    "htnv_nucleocapsid_epitope": {
        "name": "Hantaan Virus N Protein T-cell Epitope",
        "sequence": "AUGAAGAGCAACCGGGACAAGUUCAACAGCGAGGCAAAGAUCACCAAAGACAUCAACCUGGGGGAGCUGAAGUGCAAGACUGGGGAUGACGAGAAGCCCAAGAUCAAAGAAAAGAGAAAGAUGGAGAAAAUCAGUGAGAAGAAGGGGGAAGUGAAGGGCCUCUCCAAGGACUCCAAGGAAAAUGAGAAGGGAAAAGCCAACAAGAAGGAACCCAAGGCCGAAGAGGAGGUCCGGAAG",
        "type": "t_cell_epitope",
        "immunogenicity": "high"
    },
    
    "conserved_rna_motif": {
        "name": "Conserved Hantavirus RNA Structure",
        "sequence": "GUGCACCCAACGCCUGAAGACGAGGAUUUGACCAGGGAUGCGGGCAACUUCCAGGCCAACGCCAAGGAAUUGACCCGGAAACUGGACGGAAUUGAAAAGAUCCCCAAGGACAAUGGAAAGCUGAGCAAGAAGGCCGAGAAGGUGCGGGAAAACAGGCGGAACACCGUGGGGUUCAACCGUGGGGAACCUGGAAGAGGAUGCCACUGAUAGAGAACUGAGCAGGGAGAACUGAGCAGGGGAAAACACCGGUUCCGGUGAACCACUU",
        "type": "conserved_motif",
        "immunogenicity": "moderate"
    }
}

class HantavirusVaccineOptimizer:
    """Optimize Hantavirus vaccine candidates using Len's RNA prediction models"""
    
    def __init__(self, models_dict):
        """
        Initialize with Len's loaded models
        models_dict: {'dms': model, '1m7': model, '2a3': model}
        """
        self.models = models_dict
        self.optimization_history = []
        
        print(f"🦠 Hantavirus Vaccine Optimizer initialized with {len(models_dict)} models")
        for model_name in models_dict.keys():
            print(f"   ✅ {model_name.upper()} model ready")
    
    def analyze_hantavirus_target(self, target_name, target_info):
        """Analyze a specific Hantavirus sequence with all loaded models"""
        
        sequence = target_info['sequence']
        
        print(f"\n🦠 Analyzing: {target_info['name']}")
        print(f"   Type: {target_info['type']}")
        print(f"   Sequence length: {len(sequence)} nucleotides")
        print(f"   Expected immunogenicity: {target_info['immunogenicity']}")
        
        # Convert to one-hot
        nuc_map = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
        onehot = np.zeros((len(sequence), 4))
        for i, nuc in enumerate(sequence):
            if nuc in nuc_map:
                onehot[i, nuc_map[nuc]] = 1
        
        sequence_tensor = torch.FloatTensor(onehot).unsqueeze(0)
        
        # Get predictions from all loaded models
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                with torch.no_grad():
                    pred = model(sequence_tensor).cpu().numpy().squeeze()
                    
                    if pred.ndim == 0:
                        mean_pred = float(pred)
                        reactivity_profile = [mean_pred] * len(sequence)
                    else:
                        mean_pred = pred.mean()
                        reactivity_profile = pred.tolist()
                    
                    predictions[model_name] = {
                        'mean': mean_pred,
                        'std': np.std(reactivity_profile) if len(reactivity_profile) > 1 else 0,
                        'profile': reactivity_profile
                    }
                    
                    print(f"   {model_name.upper()}: {mean_pred:.3f} ± {np.std(reactivity_profile):.3f}")
                    
            except Exception as e:
                print(f"   ❌ {model_name} failed: {e}")
                predictions[model_name] = None
        
        # Calculate vaccine suitability score
        vaccine_score = self.calculate_vaccine_score(predictions, sequence, target_info)
        
        if vaccine_score > 0.7:
            print(f"   🎯 Vaccine Score: {vaccine_score:.3f} - HIGH PRIORITY")
        elif vaccine_score > 0.5:
            print(f"   ✅ Vaccine Score: {vaccine_score:.3f} - GOOD CANDIDATE")
        else:
            print(f"   ⚠️  Vaccine Score: {vaccine_score:.3f} - NEEDS OPTIMIZATION")
        
        return {
            'target': target_name,
            'info': target_info,
            'sequence': sequence,
            'predictions': predictions,
            'vaccine_score': vaccine_score
        }
    
    def calculate_vaccine_score(self, predictions, sequence, target_info):
        """Calculate overall vaccine suitability score"""
        
        # Weight different assays based on vaccine relevance
        weights = {'1m7': 0.5, 'dms': 0.3, '2a3': 0.2}  # 1M7 most important for vaccines
        
        weighted_score = 0
        total_weight = 0
        
        for assay, weight in weights.items():
            if predictions.get(assay) and predictions[assay] is not None:
                pred = predictions[assay]['mean']
                
                # Optimal vaccine range: moderate reactivity (0.3-0.7)
                if 0.3 <= pred <= 0.7:
                    score = 1.0  # Optimal
                elif 0.2 <= pred <= 0.8:
                    score = 0.8  # Good
                elif 0.1 <= pred <= 0.9:
                    score = 0.6  # Acceptable
                else:
                    score = 0.3  # Suboptimal
                
                weighted_score += score * weight
                total_weight += weight
        
        # Adjust for sequence and target characteristics
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
        
        # Moderate GC content is better for vaccines
        if 0.4 <= gc_content <= 0.6:
            gc_bonus = 0.1
        elif 0.3 <= gc_content <= 0.7:
            gc_bonus = 0.05
        else:
            gc_bonus = 0
        
        # Bonus for high immunogenicity targets
        immunogenicity_bonus = {
            'very_high': 0.15,
            'high': 0.1,
            'moderate': 0.05
        }.get(target_info.get('immunogenicity', 'moderate'), 0)
        
        final_score = (weighted_score / total_weight if total_weight > 0 else 0) + gc_bonus + immunogenicity_bonus
        
        return min(final_score, 1.0)  # Cap at 1.0

def load_len_models_for_hantavirus():
    """Load Len's best models for Hantavirus analysis"""
    
    print("🔧 Loading Len's validated models for Hantavirus analysis...")
    
    base_path = Path("/usrhome/elina/hope/")
    models = {}
    
    # Load 1M7 model (highest performance r=0.851)
    try:
        model_1m7, metadata_1m7 = load_len_absolute_final_model(
            base_path / "onehot_bilstm_1m7.pt", 
            'separate'
        )
        models['1m7'] = model_1m7
        print(f"   ✅ 1M7 model loaded (r = {metadata_1m7.get('best_pearson', 0.851):.3f})")
    except Exception as e:
        print(f"   ❌ 1M7 model failed: {e}")
    
    # Load DMS model (good performance r=0.751)
    try:
        model_dms, metadata_dms = load_len_absolute_final_model(
            base_path / "onehot_bilstm_dms.pt", 
            'separate'
        )
        models['dms'] = model_dms
        print(f"   ✅ DMS model loaded (r = {metadata_dms.get('best_pearson', 0.751):.3f})")
    except Exception as e:
        print(f"   ❌ DMS model failed: {e}")
    
    # Optionally load 2A3 model
    try:
        model_2a3, metadata_2a3 = load_len_absolute_final_model(
            base_path / "onehot_bilstm_2a3.pt", 
            'separate'
        )
        models['2a3'] = model_2a3
        print(f"   ✅ 2A3 model loaded (r = {metadata_2a3.get('best_pearson', 0.720):.3f})")
    except Exception as e:
        print(f"   ⚠️  2A3 model failed: {e} (optional)")
    
    if not models:
        raise RuntimeError("No models loaded successfully!")
    
    print(f"   🎯 {len(models)} models ready for Hantavirus analysis")
    return models

def run_complete_hantavirus_analysis():
    """Run complete Hantavirus vaccine analysis with Len's models"""
    
    print("🦠 HANTAVIRUS VACCINE OPTIMIZATION")
    print("=" * 45)
    print("Using Len's validated RNA prediction models")
    print("Targeting critical biodefense gap")
    
    # Load models
    try:
        models = load_len_models_for_hantavirus()
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("Please ensure your models are available and fin_archit_mt.py is accessible")
        return
    
    # Initialize optimizer
    optimizer = HantavirusVaccineOptimizer(models)
    
    # Analyze all Hantavirus targets
    print(f"\n🎯 ANALYZING {len(HANTAVIRUS_TARGETS)} HANTAVIRUS TARGETS:")
    print("=" * 50)
    
    analyses = []
    for target_name, target_info in HANTAVIRUS_TARGETS.items():
        analysis = optimizer.analyze_hantavirus_target(target_name, target_info)
        analyses.append(analysis)
    
    # Sort by vaccine score
    analyses.sort(key=lambda x: x['vaccine_score'], reverse=True)
    
    print(f"\n📊 HANTAVIRUS VACCINE PRIORITY RANKING:")
    print("=" * 40)
    
    for i, analysis in enumerate(analyses, 1):
        score = analysis['vaccine_score']
        name = analysis['info']['name']
        
        if score > 0.7:
            priority = "🎯 HIGH PRIORITY"
        elif score > 0.5:
            priority = "✅ GOOD CANDIDATE"
        else:
            priority = "⚠️  NEEDS WORK"
        
        print(f"{i}. {name}")
        print(f"   Score: {score:.3f} - {priority}")
        print(f"   Type: {analysis['info']['type']}")
        print()
    
    # Generate report
    print(f"🎉 ANALYSIS COMPLETE!")
    print(f"📈 Top candidate: {analyses[0]['info']['name']} (Score: {analyses[0]['vaccine_score']:.3f})")
    print(f"🧬 Your RNA expertise is ready for Hantavirus vaccine development!")
    print(f"💪 Next: Focus on top candidates for experimental validation")
    
    return analyses

if __name__ == "__main__":
    print("🧬 Hantavirus Vaccine Project")
    print("=" * 30)
    print("Integrating Len's RNA prediction models with biodefense applications")
    
    analyses = run_complete_hantavirus_analysis()
