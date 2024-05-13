# Injecting Graph Knowledge into multilingual LLMs for facilitating LRLs via Adapters

**Project Overview:**

This repository houses the code and resources for the research project titled "Injecting Commonsense Knowledge into Multilingual Large Language Models (mLLMs)" aiming to enhance the performance of low-resource languages (LRLs) through the integration of common sense knowledge from linguistic ontologies.

**Abstract:**

This project builds upon recent advancements in language model adaptation, particularly inspired by works like K-ADAPTER (Wang et al., 2020) and MAD-X (Pfeiffer et al., 2020). The primary focus is on injecting knowledge from ConceptNet and Wikipedia into mLLMs for languages classified as low-resource (Maltese, Bulgarian, and Indonesian) according to Joshi et al. (2020).

**Key Objectives:**
1. **Integration of External Knowledge:** Develop a novel approach to inject structured knowledge from ConceptNet and Wikipedia into mLLMs through the use of language adapters.
  
2. **Impact Assessment:** Evaluate the impact of injected external graph-like knowledge on mLLMs by employing various objectives, including normal Masked Language Model (MLM), MLM with targeted masking, and MLM with relationship type masking.

3. **Task-Specific Adapters:** Train task-specific adapters stacked on top of language adapters for downstream tasks such as Sentiment Analysis (SA), Named Entity Recognition (NER), and Part-of-Speech (PoS) tagging.

4. **Empirical Evaluation:** Conduct empirical evaluations on language-specific tasks to analyze the effects of structured common sense knowledge on the performance of LRLs, providing insights into the potential advantages of adapting mLLMs for low-resource scenarios.

**Dataset Information:**
The datasets used to train language adapters can be found in [this HuggingFace repository](https://huggingface.co/datasets/DGurgurov/language_adapters_data).

**Pre-trained Models // Language Adapters // Misc Models:**
Explore pre-trained language adapters models for low-resource languages in [this HuggingFace repository](https://huggingface.co/datasets/DGurgurov/language_adapters).

**How to Use:**
- The repository contains code for injecting knowledge, training language adapters, and evaluating performance on downstream tasks. Refer to the documentation and code comments for detailed instructions. To be extended.

**Citation:**
- If you use this work in your research, please cite our paper (citation details to be added).

**License:**
- This project is licensed under [MIT]. See the [LICENSE](LICENSE) file for details.

**Acknowledgments:**
- We acknowledge the contributions of the open-source community and the datasets used in this research.

**Contact:**
- For inquiries, reach out to [Daniil Gurgurov] at [dgurgurov@gmail.com].
