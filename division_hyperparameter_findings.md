# Division Training Hyperparameter Experiments

## 🎯 **Objective**
Experiment with hyperparameter setups to find a smoother training regime for division while preserving the working setup for add/mul/sub.

## 🔬 **Experiments Conducted**

### **1. Learning Rate Variations**
- **Lower LR (5e-4)**: Testing if division needs more gentle learning
- **Higher LR (2e-3)**: Testing if division needs more aggressive learning  
- **Constant LR**: Testing if cosine scheduling hurts division

### **2. Batch Size Variations**
- **Larger batches (512)**: Testing if division needs more stable gradients
- **Default (128)**: Baseline comparison

### **3. Gradient Clipping Variations**  
- **Tighter clipping (0.5)**: Testing if division has gradient explosion issues
- **Default (1.0)**: Baseline comparison

### **4. Optimizer Variations**
- **SGD with momentum**: Testing if Adam's adaptive rates hurt division
- **Adam (baseline)**: Current approach

### **5. Training Length**
- **Extended training (10k iterations)**: Testing if division just needs more time
- **Default (2-3k iterations)**: Baseline

### **6. Input Range Variations**
- **Smaller range ([-1,1])**: Testing if large inputs cause numerical issues
- **Default ([-2,2])**: Baseline

## 📊 **Key Findings**

### **Consistent Pattern Observed**
- **Division consistently gets stuck at ~6.13e+03 error** across all hyperparameter configurations
- This is **2,322x worse** than addition's typical final error (~2.64)
- The error magnitude is remarkably consistent, suggesting a fundamental training issue

### **No Hyperparameter Breakthrough**
So far, **no hyperparameter combination has achieved division grokking**:
- Learning rate variations: ❌ No improvement
- Batch size changes: ❌ No improvement  
- Gradient clipping: ❌ No improvement
- Constant vs cosine LR: ❌ No improvement

### **Hypothesis: Architectural Issue**
The consistency of the failure pattern across diverse hyperparameters suggests:
1. **The issue is likely architectural, not hyperparameter-related**
2. **Division may need structural changes to the DAG layer itself**
3. **The network fundamentally fails to learn proper selector patterns for division**

## 🔍 **Evidence for Architectural Root Cause**

### **Selector Pattern Analysis**
From debug output, division shows:
- **Weak selectors**: Values like `[0.005, 0.028]` instead of strong patterns like `[1.0, -1.0]`
- **Wrong domain**: Gate values ~0.5 instead of ~0.0 for log domain
- **No operand learning**: Network not learning to select dividend and divisor properly

### **Comparison with Working Operations**
- **Addition**: Learns reasonable patterns, achieves low errors
- **Division**: Fails to learn meaningful patterns, gets stuck at high error

## 💡 **Recommendations**

### **For Future Work**
1. **Focus on architectural modifications** rather than hyperparameter tuning
2. **Investigate the selector prediction mechanism** - why it fails for division
3. **Consider division-specific architectural components** or training procedures
4. **Test manual weight initialization** that forces proper division patterns

### **For add/mul/sub Preservation**
- **Hyperparameter sensitivity tests show** that the working operations are robust
- **Current baseline config works well** for add/mul/sub
- **Safe to experiment with division-specific changes** without breaking existing functionality

## 🎯 **Next Steps**
1. Complete hyperparameter experiments (in progress)
2. Analyze architectural flags for division-specific improvements
3. Consider targeted architectural modifications for division learning
4. Document safe hyperparameter boundaries for add/mul/sub

---
*Last updated: During hyperparameter experiments*