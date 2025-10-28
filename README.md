# DUDA: A Two-stage Decoupling Unsupervised Domain Adaptation Framework for Semi-supervised Singing Melody Extraction from Polyphonic Music

This repository contains the official PyTorch implementation for the paper accepted by ACM MM 2025 — [DUDA: A Two-stage Decoupling Unsupervised Domain Adaptation Framework for Semi-supervised Singing Melody Extraction from Polyphonic Music](https://dl.acm.org/doi/10.1145/3746027.3755747).

In this paper, we propose a novel two-stage decoupling unsupervised domain adaptation framework for semi-supervised singing melody extraction, termed as DUDA. Specifically, in the first stage, we decouple the holistic information into fine-grained information: tone and octave, and narrow the domain gap at the tone and octave level, respectively. This enables the model to align the tone-octave information between source and target domains for better feature distribution. Then, we leverage the learned domain-agnostic fine-grained features as additional information to obtain domain-agnostic holistic features. We also suggest to align intra-domain, inter-domain, and sample-level features to further improve the performances. In the second stage, we propose a novel tone-octave consistency regularization method by leveraging the extracted fine-grained information to judge the availability of unlabeled data.

<img src="assets/duda.png" width="800px"/>
