# Injecting Graph Knowledge into multilingual LLMs for facilitating LRLs via Adapters

**Project Overview:**

This repository houses the code and resources for the research project titled "Adapting Multilingual LLMs to Low-Resource Languages with Knowledge Graphs via Adapters" aiming to enhance the performance of low-resource languages (LRLs) through the integration of graph knowledge from linguistic ontologies.

**Abstract:**

This paper explores the integration of graph knowledge from linguistic ontologies into multilingual Large Language Models (LLMs) using adapters to improve performance for low-resource languages (LRLs) in sentiment analysis (SA) and named entity recognition (NER). Building upon successful parameter-efficient fine-tuning techniques, such as K-ADAPTER and MAD-X, we propose a similar approach for incorporating knowledge from multilingual graphs, connecting concepts in various languages with each other through linguistic relationships, into multilingual LLMs for LRLs. Specifically, we focus on eight LRLs —Maltese, Bulgarian, Indonesian, Nepali, Javanese, Uyghur, Tibetan, and Sinhala — and employ language-specific adapters fine-tuned on data extracted from the language-specific section of ConceptNet, aiming to enable knowledge transfer across the languages covered by the knowledge graph. We compare various fine-tuning objectives, including standard Masked Language Modeling (MLM), MLM with full-word masking, and MLM with targeted masking, to analyze their effectiveness in learning and integrating the extracted graph data. Through empirical evaluation on language-specific tasks, we assess how structured graph knowledge affects the performance of multilingual LLMs for LRLs in SA and NER, providing insights into the potential benefits of adapting language models for low-resource scenarios.

**Key Objectives:**
1. **Integration of External Knowledge:** Develop a novel approach to inject structured knowledge from ConceptNet and Wikipedia into mLLMs through the use of language adapters.
  
2. **Impact Assessment:** Evaluate the impact of injected external graph-like knowledge on mLLMs by employing various objectives, including normal Masked Language Model (MLM), MLM with targeted masking, and MLM with relationship type masking.

3. **Task-Specific Adapters:** Train task-specific adapters stacked on top of language adapters for downstream tasks such as Sentiment Analysis (SA), Named Entity Recognition (NER), and Part-of-Speech (PoS) tagging.

4. **Empirical Evaluation:** Conduct empirical evaluations on language-specific tasks to analyze the effects of structured common sense knowledge on the performance of LRLs, providing insights into the potential advantages of adapting mLLMs for low-resource scenarios.

**Architecture Overview:**

One of the Wiki or ConceptNet language adapters is used during inference. The outputs then go to a task adapter, which is followed by a classification head. If fusion is specified, the fusion mechanism is activated.

<img src="https://github.com/d-gurgurov/Injecting-Commonsense-Knowledge-into-LLMs/blob/main/assets/kallm.png?raw=true" alt="architecture" width="500"/>

**Dataset Information:**
The datasets used to train language adapters and task adapters can be found on [HuggingFace](https://huggingface.co/datasets/DGurgurov).

**Pre-trained Models // Language Adapters // Misc Models:**
Explore pre-trained language adapters models for low-resource languages in [this HuggingFace repository](https://huggingface.co/datasets/DGurgurov/).

**How to Use:**
- The repository contains code for injecting knowledge, training language adapters, and evaluating performance on downstream tasks. Refer to the documentation and code comments for detailed instructions. To be extended.

**Citation:**
- If you use this work in your research, please cite our paper:

```bibtex
@article{gurgurov2024adapting,
  title={Adapting Multilingual LLMs to Low-Resource Languages with Knowledge Graphs via Adapters},
  author={Gurgurov, Daniil and Hartmann, Mareike and Ostermann, Simon},
  journal={arXiv preprint arXiv:2407.01406},
  year={2024}
}
```

**License:**
- This project is licensed under [MIT]. See the [LICENSE](LICENSE) file for details.

**Acknowledgments:**
- We acknowledge the contributions of the open-source community and the datasets used in this research.

**Contact:**
- For inquiries, reach out to [Daniil Gurgurov] at [dgurgurov@gmail.com].
