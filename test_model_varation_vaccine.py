"""
Quick Model Validation Test
==========================

Test your loaded models with different RNA sequences to verify they show
appropriate variation and behavior.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Import your final architecture (assuming you have the models loaded)
# You can copy the model classes from absolute_final_fix.py if needed

def test_model_variation():
    """Test models with different RNA sequences"""
    
    print("🧪 TESTING MODEL VARIATION")
    print("=" * 35)
    
    # Test sequences with expected different reactivities
    test_sequences = {
        'very_structured': "GCGCGCGCGCGCGCGC",      # Should be less reactive
        'very_unstructured': "AAAUUUAAAUUUAAAA",     # Should be more reactive  
        'mixed_structure': "AGCUAGCUAGCUAGCU",       # Should be moderate
        'original_covid': "GGAAAGCCUGUCUCUGCUACCGCCGCCACCAUGGCCAAGAAGGUG",
        'high_gc': "GGGCCCGGGCCCGGGCCC",             # Very structured
        'low_gc': "AAAUUUAAAUUUAAAUUU"               # Very unstructured
    }
    
    # Nucleotide mapping for one-hot encoding
    nuc_map = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
    
    def sequence_to_onehot(sequence):
        """Convert RNA sequence to one-hot tensor"""
        onehot = np.zeros((len(sequence), 4))
        for i, nuc in enumerate(sequence):
            if nuc in nuc_map:
                onehot[i, nuc_map[nuc]] = 1
        return torch.FloatTensor(onehot).unsqueeze(0)
    
    def calculate_gc_content(sequence):
        """Calculate GC content of sequence"""
        gc_count = sequence.count('G') + sequence.count('C')
        return gc_count / len(sequence)
    
    # You'll need to load your models here
    # Example loading (adjust paths as needed):
    from fin_archit_mt  import LenAbsoluteFinalSeparateModel, load_len_absolute_final_model
    
    base_path = Path("/usrhome/elina/hope/")
    
    # Load one model for testing (DMS model as example)
    dms_model = None
    try:
        dms_path = base_path / "onehot_bilstm_dms.pt"
        if dms_path.exists():
            dms_model, metadata = load_len_absolute_final_model(dms_path, 'separate')
            print(f"✅ Loaded DMS model (r = {metadata.get('best_pearson', 'unknown'):.3f})")
        else:
            print("⚠️  DMS model file not found")
    except Exception as e:
        print(f"❌ Error loading DMS model: {e}")
    
    if dms_model is None:
        print("❌ No model loaded for testing")
        return
    
    # Test all sequences
    results = {}
    
    print(f"\n📊 TESTING SEQUENCES:")
    print("-" * 25)
    
    for name, sequence in test_sequences.items():
        print(f"\n🔬 {name}:")
        print(f"   Sequence: {sequence}")
        print(f"   Length: {len(sequence)} nt")
        print(f"   GC content: {calculate_gc_content(sequence):.1%}")
        
        try:
            # Convert to tensor
            onehot_tensor = sequence_to_onehot(sequence)
            
            # Get prediction
            with torch.no_grad():
                pred = dms_model(onehot_tensor).cpu().numpy().squeeze()
                
                if pred.ndim == 0:  # Single value
                    mean_reactivity = float(pred)
                    std_reactivity = 0.0
                else:  # Multiple values
                    mean_reactivity = pred.mean()
                    std_reactivity = pred.std()
                
                results[name] = {
                    'mean': mean_reactivity,
                    'std': std_reactivity,
                    'gc_content': calculate_gc_content(sequence)
                }
                
                print(f"   Mean reactivity: {mean_reactivity:.3f}")
                print(f"   Std reactivity: {std_reactivity:.3f}")
                
                # Quick interpretation
                if mean_reactivity > 0.7:
                    print(f"   📈 High reactivity")
                elif mean_reactivity < 0.3:
                    print(f"   📉 Low reactivity") 
                else:
                    print(f"   📊 Moderate reactivity")
                    
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Analysis
    if len(results) >= 3:
        print(f"\n📈 VARIATION ANALYSIS:")
        print("-" * 20)
        
        all_means = [r['mean'] for r in results.values()]
        overall_range = max(all_means) - min(all_means)
        overall_std = np.std(all_means)
        
        print(f"Overall range: {overall_range:.3f}")
        print(f"Overall std: {overall_std:.3f}")
        
        if overall_range > 0.1:
            print(f"✅ Good variation! Model shows sequence-dependent behavior")
        elif overall_range > 0.05:
            print(f"🟡 Moderate variation")
        else:
            print(f"⚠️  Low variation - might be saturation issue")
        
        # Check GC content correlation
        gc_contents = [r['gc_content'] for r in results.values()]
        correlation = np.corrcoef(all_means, gc_contents)[0,1]
        print(f"GC-reactivity correlation: {correlation:.3f}")
    
    return results

if __name__ == "__main__":
    print("🧬 Model Validation Test")
    print("=" * 25)
    print("Testing RNA sequence variation...")
    
    results = test_model_variation()
    
    if results:
        print(f"\n🎯 SUMMARY:")
        print("Your model is loaded and making predictions!")
        print("Check if different sequences give different reactivities.")
        print("\nExpected behavior:")
        print("- Structured sequences (high GC) → lower reactivity")
        print("- Unstructured sequences (low GC) → higher reactivity") 
    else:
        print("\n⚠️  No results - check model loading")
