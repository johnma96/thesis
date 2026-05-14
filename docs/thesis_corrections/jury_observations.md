# Jury observations — Master's thesis

**Evaluator:** Manuel Mauricio Goez Mora (ITM)  
**Date:** April 13, 2026  
**Verdict:** Conditionally approved — corrections required within 3 weeks

All 20 observations were addressed and the thesis was approved in May 2026.

---

## Category A — Formatting and editing

1. **Decimal separators:** Normalize decimal separators throughout the document (text, figures, and tables).

2. **Spelling and typos:** Correct spelling errors and typos identified in the annotated document.

3. **HSI definition redundancy:** The definition of Hyperspectral Imaging (HSI) appears in three or more sections. Unify into a single canonical definition with cross-references.

---

## Category B — Written clarifications and justifications

4. **NDVI threshold justification:** Justify the NDVI threshold used for the vegetation mask. Include literature ranges and variations explored.

5. **High wellness criterion:** Analyze whether the "high wellness" criterion excludes severely deficient plants from the dataset.

6. **Band selection methodology:** Expand the description of the band selection technique and explain why other dimensionality reduction methods were not used.

7. **Validation/test split:** Justify that the validation and test sets come from the same spatial sample.

8. **Single aerial capture:** Clarify that the study used a single aerial capture, not two.

9. **Twelve initial algorithms:** Justify the selection of 12 initial algorithms versus the more than 20 mentioned in §2.3.2.

10. **Random Forest analysis:** Deepen the analysis of Random Forest given its high baseline performance shown in Table 4-1.

11. **Hyperparameter tuning gains:** Explain why hyperparameter optimization produced no significant gains in three of the final ML models.

12. **FP vs FN agronomic context:** Discuss false positives versus false negatives in the agronomic context of phosphorus deficiency detection.

13. **Mixed-stress parcels:** Analyze the practical case of parcels with mixed stressed/non-stressed plants and its effect on CNN-2D performance.

14. **Conclusions restructure:** Restructure conclusions to provide evidence for each specific objective, and include limitations and justification of the binary vs. multiclass classification choice.

15. **Future work:** Validate the approach against other biotic/abiotic stresses and consider integrating genotype information.

---

## Category C — Additional analysis using existing results

16. **Confusion matrices in percentages:** Convert confusion matrices to percentages to better visualize the improvement of CNN-2D. Clarify the reduction in data volume for the CNN-2D matrix.

    > *"se propone cambiar las matrices de confusión por porcentajes ya que permite identificar la mejora significativa del modelo CNN-2D ya que este modelo presenta menor cantidad de datos, también es importante aclarar al lector que no esté familiarizado [el] motivo de la reducción de datos en la matriz de confusión."*

17. **Vegetation index validation:** Validate the real influence of the calculated vegetation indices versus the spectral features used in the modeling process.

    > *"validar la influencia real de los índices de vegetación calculados frente a las características utilizadas en el proceso"*

18. **Computational cost metrics:** Add computational cost metrics for the DL models (training time, memory usage, inference latency).

19. **Experimental design diagram:** Add a diagram illustrating the experimental design.

---

## Category D — Critical methodological concern (highest priority)

20. **CNN-2D exceptionally high PR-AUC (0.963):** The jury considers this result exceptionally high and suspects the network may be learning the spatial structure of the manually labeled polygons rather than the spectral signature of stress.

    Sub-tasks:
    - **20a:** Was genotype used as a variable? Was performance evaluated per variety? (8 genotypes with naturally distinct spectral profiles)
    - **20b:** Does the train/val/test split mix genotypes across folds? Possible data leakage by variety.
    - **20c:** Write a defensive analysis and discussion section.

    > *Resolved:* Per-genotype analysis and two spectral ablation probes confirmed that the model learns the spectral fingerprint of stress, not polygon geometry. Full analysis in `notebooks/401-jmmz-genotype-analysis.ipynb` and `notebooks/402-jmmz-ablation-tests.ipynb`.
