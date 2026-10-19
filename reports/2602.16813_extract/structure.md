# Structure of 2602.16813v3.pdf

- Pages: 62; extracted characters: 214235
- References start on page 20 (body is pages 1-20; later pages are references/appendix)

## First lines of page 1 (title / authors / affiliations)

    Flow Map Language Models:
    One-step Language Modeling via Continuous Denoising
    Chanhyuk Lee1 , Jaehoon Yoo1 , Manan Agarwal2 , Sheel Shah2 , Jerry Huang2
    Aditi Raghunathan2 , Seunghoon Hong1
    Nicholas M. Boffi†,2 , Jinwoo Kim†,1
    1
    KAIST,
    2
    Carnegie Mellon University
    Equal advising

## Abstract (auto-detected, verify against paper.txt)

Language models based on discrete diffusion have attracted widespread interest for their potential to provide faster generation than autoregressive models. Despite their promise, these models typically produce samples whose quality sharply degrades in the few-step regime, preventing a dramatic speedup in practice. Here, we show that language models based on continuous flows over one-hot token embeddings can outperform discrete diffusion in both quality and speed. Importantly, our continuous formulation defines a unique flow map that can be learned directly for efficient few-step inference, a structure we show is unavailable to discrete methods. In this setting, we show that both the flow and its associated flow map can be learned with simple cross-entropy objectives that respect the simplex geometry of the data, and we identify three distinct choices for flow map distillation whose performance we compare in practice. Using these insights, we build a flow language model (FLM), a continuous flow that matches state-of-the-art discrete diffusion baselines on the One Billion Words (LM1B) and OpenWebText (OWT) datasets. We then distill FLM into a flow map language model (FMLM), whose one-step generation exceeds the 8-step quality of recent few-step discrete diffusion language models. Our work challenges the widely-held hypothesis that discrete noising processes are necessary for generative modeling over discrete modalities and paves the way toward accelerated language modeling at scale. Code: https://github.com/david3684/flm

## Headings (page: heading)

- p.1: Introduction
- p.3: Background
- p.11: Experiments
- p.16: Method
- p.16: 8 Steps
- p.16: 4 Steps
- p.16: 2 Steps
- p.17: Ablation Study
- p.19: Conclusion
- p.19: Acknowledgments
- p.20: References
- p.27: Related Work
- p.42: E D̂s (Is ) +
- p.46: 1 Steps 2 Steps
- p.46: 4 Steps 1024 Steps
- p.56: I I f not to me <|endoftext|>

## Figure and table captions

- p.1 **Figure 1**: Flow map language models. Our FMLM outperforms discrete diffusion models (gray) and matches the 8-step generation performance of distilled discrete diffusion models (light purple) in only one step (dark purple).
- p.2 **Figure 2**: Overview. (Left) We leverage a simple continuous interpolation between Gaussian noise and a one-hot encoding of language data. (Middle) Our FLM learns a denoiser that predicts the posterior over clean data, which we convert into a flow for sampling. (Right) Ou
- p.3 **Figure 3**: Factorization error in discrete diffusion. A toy dataset with two correlated modes new-york and san-diego. (Left) In many-step sampling, both continuous flows and discrete diffusion models generate valid data. (Right) With few-step sampling, the factorized tra
- p.10 **Figure 9**: Decoding error rate. Our time reparameterization τ (t) redistributes time so each step Distillation. Following Section 3.3, to obtain the flow contributes uniformly to the denoising signal; time map we learn the two-time denoiser δ̂s,t defined by (18). Followi
- p.12 **Table 10**: FLM Performance. Generation performance of FLM at 1024 sampling steps in comparison to discrete diffusion baselines. Ground truth dataset entropy shown in parentheses. Our approach attains state-of-the-art generative perplexity while maintaining entropy close 
- p.12 **Figure 11**: FLM generation quality. Generation performance of FLM on LM1B (left) and OWT (right) compared to diffusion baselines. FLM outperforms baselines at large step counts. Its performance degrades at low step counts, as it has not yet been distilled into an FMLM. Tr
- p.13 **Figure 12**: FMLM few-step generation. Few-step generation performance of FMLM on LM1B (left) and OWT (right) compared to distilled discrete diffusion. FMLM maintains strong generative perplexity across step counts and achieves state-of-the-art performance in the very few-
- p.13 **Table 13**: FMLM performance. Comparison between FMLM and baseline few-step distilled discrete diffusion models in the extreme few-step regime. Similar to the many-step regime, our model attains state of the art performance and while maintaining the entropy of the generat
- p.14 **Figure 14**: Qualitative one-step generation. One-step samples from FMLM and distilled discrete diffusion baselines trained on LM1B. FMLM produces coherent, grammatical text, while discrete diffusion baselines generate incoherent token sequences (red, Gen. PPL > 1000) or r
- p.16 **Figure 16**: Autoguidance stability. FLM maintains stable generation quality across guidance scales η up to 100, while discrete baselines fail at η ≥ 10. Shaded region shows Gen. PPL > 1000 or entropy < 3.9, indicating nonsensical or collapsed generation. Results shown on 
- p.16 **Table 17**: Reward-guided generation. We apply FMRG [50] to steer generation toward four target properties (topic, grammaticality, sentiment, safety) across varying step counts on OWT. At 1024 steps, the undistilled teacher models (MDLM, Duo) are used; at fewer steps, the
- p.17 **Table 18**: Ablation results on the LM1B dataset. * denotes 300k training steps. All embedding ablation models use learned time reparameterization from Dieleman et al. [13]. FLM, 1024-step generation Configuration
- p.18 **Figure 19**: Valid one-step samples Figure 20: Invalid one-step samples Figure 21: Valid one-step samples from FMLM. from FMLM. from FMLM conditioned on clues .
- p.18 **Table 22**: Sudoku generation and solving validity (% among 1024 samples). Unconditional
- p.39 **Figure 23**: Eulerian and Lagrangian objectives on the simplex. (Left) The Eulerian teacher δ̄s,t is constructed from D̂s (Is ) and derivatives of δ̂s,t . (Right) The Lagrangian teacher is constructed from D̂t (X̂s,t (Is )), requiring an intermediate flow map evaluation of
- p.43 **Table 24**: Generation performance of FMLM trained with cross-entropy distillation (Sections 3.3 and 4) and two-stage squared-loss distillation (Appendix D) in the extreme few-step regime. LM1B
- p.45 **Table 25**: Self-BLEU [60] score of 1,024 generated samples in the one-step generation setting. Lower score denotes more n-gram diversity. For reference, we report the Self-BLEU score of mode-collapsed case when all the samples are identical, and the score of the referenc
- p.45 **Table 26**: LLM-based win rate (%) measured by GPT-4.1 as a judge. Win-rate of 0.5 denotes the data distribution has equal amount of diversity compared with real data, when measured by LLM. Dataset
- p.45 **Table 27**: Performance of FLM trained on OWT for 150k steps, across different model sizes. Model Size
- p.46 **Table 28**: Performance of FMLM with uniform and Gaussian priors trained on LM1B dataset for 200k steps. Prior
- p.46 **Table 29**: Uniqueness of generated Sudoku grids. Model
- p.46 **Table 30**: Novelty of generated Sudoku grids.
- p.46 **Figure 31**: Ablation on flow map characterization. Training loss and generation quality (Gen. PPL) of different flow map distillation objectives. Scaling behavior of FLM. To investigate the scaling behavior of FLM, we evaluate performance across varying base model capacit
- p.47 **Table 32**: Memory and time cost per training step. Peak GPU memory (GB) and wall-clock time (seconds) per training step for FLM and discrete diffusion baselines, measured across vocabulary sizes |V | ∈ {30k, 50k} and sequence lengths L ∈ {128, 1024}.
- p.49 **Figure 33**: Qualitative one-step conditional generation. One-step conditional samples from FMLM and distilled discrete diffusion baselines trained on OWT. Generated text are colored. While FMLM generates grammarly correct sentence highly correlated with the given prefix, 
- p.50 **Figure 34**: Samples generated by FLM trained on LM1B with different sampling steps.
- p.51 **Figure 35**: Samples generated by FLM trained on OWT with different sampling steps. 51
- p.52 **Figure 36**: One-step samples generated by FMLM trained on LM1B.
- p.53 **Figure 37**: One-step conditional samples generated by FMLM trained on OWT. Generated text are colored.
- p.54 **Figure 38**: One-step samples generated by FMLM trained on OWT.
- p.55 **Figure 39**: One-step samples generated by few-step masked discrete diffusion baselines trained on OWT. 55
- p.56 **Figure 40**: One-step samples generated by few-step uniform discrete diffusion baselines trained on OWT. 56
- p.57 **Figure 41**: Samples from FMLM trained on LM1B from fixed starting noise and varying the number of steps.
- p.58 **Figure 42**: Samples from FMLM trained on OWT from fixed starting noise and varying the number of steps. 58
- p.59 **Figure 43**: Samples generated by MDLM + SDTT [10] trained on LM1B from fixed initial random seed and varying the number of sampling steps.
- p.60 **Figure 44**: Samples generated by Duo + DCD [19] trained on LM1B from fixed initial random seed and varying the number of sampling steps.
- p.61 **Figure 45**: A sample from FMLM+FMRG (Section 5.3), rewarded by safety (TweetVal-Offensive [64], Label=Non-offensive).
- p.62 **Figure 46**: A sample from FMLM+FMRG (Section 5.3), rewarded by topic (AG News [61], Label=Sports).

## Numbered equations seen (equation number, page)

- Eq. (1) p.44

## Rendered pages (view with Read)

- reports/2602.16813_extract/pages/page-01.png
- reports/2602.16813_extract/pages/page-02.png
- reports/2602.16813_extract/pages/page-03.png
- reports/2602.16813_extract/pages/page-10.png
- reports/2602.16813_extract/pages/page-12.png
- reports/2602.16813_extract/pages/page-13.png
- reports/2602.16813_extract/pages/page-14.png
- reports/2602.16813_extract/pages/page-16.png
- reports/2602.16813_extract/pages/page-18.png
- reports/2602.16813_extract/pages/page-39.png
