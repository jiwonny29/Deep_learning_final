# Adversarial Patch Attacks on CIFAR-10

## Overview
Adversarial patch attacks exploit vulnerabilities in deep learning models by creating universal perturbations that consistently mislead classifiers. Unlike traditional adversarial examples, these patches are large, image-independent, and effective across transformations like scaling and rotation. This study adapts the methodology proposed by Brown et al. to the CIFAR-10 dataset, analyzing factors such as:

- **Patch size**
- **Untargeted Success Rate**
- **Targeted Success Rate**
- **Transferability across models** (e.g., ResNet-18, DenseNet, VGG, MobileNet, EfficientNet)
- **Multiple Small Patches**

The findings demonstrate the risks these patches pose to deep learning systems, highlighting the need for robust defenses.

## Key Contributions
1. **Patch Size Impact**: Larger patches achieve higher untargeted Attack Success Rate (ASR), with success rates of up to **86.31%**.
2. **Class Vulnerabilities**: Targeted attacks show that classes like `Car`, `Truck`, and `Bird` are particularly susceptible, while `Deer` and `Frog` remain resilient.
3. **Model Transferability**: Patches demonstrate robustness across architectures, with **VGG** and **DenseNet** showing the highest vulnerability.
4. **Multiple Patches and Data Transformations**: Trade-offs between subtlety, patch size, and ASR are explored, revealing diminishing returns beyond a certain number of patches.

## Methodology
### 1. Patch Generation
This study builds on Brown et al.'s work, generating **rectangular adversarial patches** and training them using:
- **CrossEntropy Loss** for targeted and untargeted attacks.
- **Adam Optimizer** with a learning rate of 0.0001.

Patches were initialized randomly, placed at random positions on **32x32 CIFAR-10** images, and optimized iteratively.

### 2. Experimental Setup
- **Dataset**: CIFAR-10
- **Models**: ResNet-18, DenseNet, VGG, MobileNet, EfficientNet
- **Patch Sizes**: (3x3), (5x5), (7x7), (16x16)
- **Metrics**:
  - **Attack Success Rate (ASR)**: Proportion of images misclassified due to the adversarial patch.
  - **Targeted vs Untargeted ASR**: Success for specific target classes vs general misclassification.

### 3. Evaluation Metrics
- **Untargeted ASR**: Measures overall misclassification rates.
- **Targeted ASR**: Tracks misclassification into a specific class.
- **Transferability**: Evaluates patch robustness across ResNet-18, DenseNet, VGG, MobileNet, and EfficientNet.

## Results
### 1. Effect of Patch Size on Untargeted ASR
Smaller patches, such as (3,3), achieved an ASR of approximately 50–55%, while larger patches like (16,16) significantly improved the ASR, reaching around 75–80%.

### 2. Targeted ASR Analysis
Targeted attacks reveal class-specific vulnerabilities. Notably:
- `Car` and `Dog` classes exhibit higher ASR.

| Target Class | Size 3 | Size 5 | Size 7 | Size 16 | Average |
|--------------|--------|--------|--------|---------|---------|
| Plane        | 3.28   | 3.35   | 3.40   | 3.65    | 3.42    |
| Car          | 16.50  | 17.40  | 17.71  | 18.76   | 17.59   |
| Bird         | 2.29   | 2.20   | 1.97   | 2.59    | 2.26    |
| Cat          | 10.25  | 10.12  | 9.88   | 9.52    | 9.94    |
| Deer         | 9.41   | 10.23  | 10.49  | 12.13   | 10.57   |
| Dog          | 11.20  | 10.66  | 10.79  | 13.67   | 11.58   |
| Frog         | 9.37   | 9.16   | 9.86   | 9.35    | 9.44    |
| Horse        | 8.98   | 9.17   | 9.60   | 8.50    | 9.06    |
| Ship         | 10.14  | 9.87   | 10.01  | 14.20   | 11.05   |
| Truck        | 10.79  | 11.06  | 10.37  | 12.33   | 11.14   |
| **Average**  | 9.22   | 9.32   | 9.41   | 10.47   | 10.61   |


### 3. Patch Transferability
Adversarial patches generated on ResNet-18 transfer effectively to other models. **VGG** and **DenseNet** show the highest untargeted ASR, while MobileNet exhibits resilience:

| Transfer to  | Size 3  | Size 5  | Size 7  | Size 16 | Average |
|--------------|---------|---------|---------|---------|---------|
| DenseNet     | 71.37   | 72.42   | 77.42   | 87.26   | 77.12   |
| VGG          | 82.73   | 74.85   | 71.80   | 85.52   | 78.73   |
| MobileNet    | 43.69   | 51.06   | 61.11   | 82.62   | 59.62   |
| EfficientNet | 60.12   | 66.44   | 74.11   | 89.84   | 72.63   |
| **Average**  | 64.48   | 66.19   | 71.11   | 86.31   | 72.03   |

### 4. Additional Experiments
- **Dataset Transformations**: Simplifying transformations improved model resilience, reducing Targeted ASR.
- **Multiple Patches**: Using multiple small patches (e.g., (5x5)) increases ASR up to **78.96%** but plateaus beyond three patches.

## Related Works
The foundational work by Brown et al. ([Adversarial Patch](https://arxiv.org/abs/1712.09665)) introduced universal, printable adversarial patches that exploit classifier vulnerabilities. This study expands on their findings by exploring patch size, class susceptibility, and transferability.

## Limitations and Future Work
### Limitations:
- Limited to **CIFAR-10** dataset and relatively simple models.
- Results may not generalize to larger datasets or complex architectures.

### Future Directions:
1. **Expand to Complex Datasets**: Evaluate on ImageNet, COCO, and real-world data.
2. **Advanced Architectures**: Test Transformer-based models and state-of-the-art classifiers.
3. **Class-Specific Analysis**: Investigate vulnerabilities across diverse target classes.
4. **Defensive Strategies**: Explore:
   - Adversarial training
   - Model ensembling
   - Advanced data augmentation techniques

## Conclusion
This study highlights the significant risks posed by adversarial patches in deep learning systems. Larger patches achieve higher ASR, certain classes are more vulnerable, and patch transferability demonstrates practical risks across architectures. These findings emphasize the urgent need for robust defense mechanisms in machine learning systems.

## References
[1] Tom Brown et al., "Adversarial Patch." [arXiv:1712.09665](https://arxiv.org/abs/1712.09665)
