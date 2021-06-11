# Integration_Segregation_Methods
Methods to measure and correct integration and segregation.

# OverallFC_methods

Some methods proposed by van den Heuvel et al [1] for "divorce" the effect of Overall FC from network topology. Methods consisted in:

* Linear regression between networks metrics and Overall FC. The slope is used to substract the Overall FC effect on topology. 
The linear regression method returns residual metrics, and difference between groups could be assesed by pairwise tests or 
by a simple permutation test (included in the scritp).
* Modified permutation test that corrects for Overall FC.

A "toy" example was provided with the script.

# HMA

Integration and Segregation quantification using hierarchical modular analysis [2,3]. Methods were adapted from: https://github.com/TobousRong/Hierarchical-module-analysis.

The methods are based in eigenmodes analysis, and not require the use of any arbitrary threshold to the functional connectivity matrices.

# References

[1] van den Heuvel, M. P., de Lange, S. C., Zalesky, A., Seguin, C., Yeo, B. T., 
& Schmidt, R. (2017). Proportional thresholding in resting-state fMRI functional 
connectivity networks and consequences for patient-control connectome 
studies: Issues and recommendations. Neuroimage, 152, 437-449.

[2] Wang, R., Lin, P., Liu, M., Wu, Y., Zhou, T., & Zhou, C. (2019). 
Hierarchical connectome modes and critical state jointly maximize 
human brain functional diversity. Physical review letters, 123(3), 
038301.

[3] Wang, R., Liu, M., Cheng, X., Wu, Y., Hildebrandt, A., & Zhou, C. (2021). 
Segregation, integration and balance of large-scale resting brain networks 
configure different cognitive abilities. arXiv preprint arXiv:2103.00475.




