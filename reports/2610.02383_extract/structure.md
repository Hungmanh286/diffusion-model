# Structure of 2610.02383.pdf

- Pages: 26; extracted characters: 72543

## First lines of page 1 (title / authors / affiliations)

    Preprint
    T HE S URPRISING E FFECTIVENESS
    OF S HARED M EMORY IN L OOPED T RANSFORMERS
    Giovanni Monea1∗ Keshav Ramji2 Yousef El-Kurdi2 Luis A. Lastras2
    Yoav Artzi1 Nathan Godey1† Ramón Fernandez Astudillo2†
    1
    Cornell University 2 IBM Research
    Looped Transformers apply the same layers several times per token, adding compute to improve quality without more parameters. Each recursion, however, writes
    its own key-value cache, so memory still grows with compute. Inference-time
    techniques can shrink this cache at a cost in quality. We pretrain looped language

## Abstract (auto-detected, verify against paper.txt)

(not detected)

## Headings (page: heading)

- p.1: T HE S URPRISING E FFECTIVENESS
- p.1: A BSTRACT
- p.2: I NTRODUCTION
- p.2: L OOPED T RANSFORMERS
- p.3: L OOPING WITH S HARED M EMORY
- p.4: M PLT
- p.4: E XPERIMENTAL S ETUP
- p.5: R ESULTS
- p.6: W HY M EMORY S HARING H ELPS
- p.8: R ELATED W ORK
- p.10: D ISCUSSION
- p.10: Acknowledgments
- p.11: R EFERENCES
- p.15: A RCHITECTURE AND T RAINING D ETAILS
- p.15: S EED VARIANCE AND S TATISTICAL T ESTS
- p.16: N ORMALIZATION BETWEEN R ECURSIONS
- p.17: A DDITIONAL A NALYSIS D ETAILS
- p.20: E XPERIMENTS AT N INE R ECURSIONS
- p.24: D ECODING S PEED AND M EMORY ON A GPU

## Figure and table captions

- p.1 **Figure 1**: Shared memory not only saves memory, it also improves quality. Left: in our Looped Prediction Transformer (LPT), later recursions read the KV cache written by the first recursion and keep only a short window of their own. Right: perplexity and context memory o
- p.4 **Table 1**: LPT and H - LPT reach lower perplexity than their looped baselines with much smaller caches. Main results across S, M, and L: validation perplexity, mean five-task accuracy (%), and context memory and FLOPs relative to same-size STD. In the STD rows, R = 1 is 
- p.5 **Figure 2**: LPT and H - LPT reach higher accuracy with much smaller caches than the looped baselines. Mean five-task accuracy vs. context memory at each size. Curves follow two, three, and five recursions, with larger markers at five. Dashed lines mark single-pass STD. wi
- p.7 **Figure 3**: LPT trains best with final loss, while LT consistently worsens with more recursions without adaptive loss. Validation perplexity (↓), collapsed runs in the upper strip on a log scale. Filled markers are final loss, hollow ones adaptive loss. S
- p.7 **Figure 4**: A short prediction window is enough. Validation perplexity of LPT relative to LT (↓) vs. window size, LT at its better objective per depth. The dotted line marks ω = 64. and Appendix D). Shared memory receives most attention at every prediction recursion in bo
- p.8 **Figure 5**: Prediction recursions attend mostly to shared memory. Attention mass assigned to shared KV entries in LPT, averaged over queries and heads. For clarity, only six layers are displayed. The bottom row averages all layers. Layer
- p.8 **Figure 6**: Shared and local keys separate in LPT. Joint query/key PCA, fitted separately for each panel. Dots are tokens and large circles their means; darker shades mark later recursions. Shared Memory Is a Gradient Highway Reading shared memory gives the final loss a d
- p.9 **Figure 7**: Shared memory restores much of the gradient reaching the first recursion. Mean gradient norm contributed by each recursion, with single-pass STD as reference. The first LPT bar is split into shared-KV (hatched) and hidden-state projections. LT is trained with 
- p.9 **Figure 8**: Detaching shared-cache gradients hurts learning. Validation perplexity (↓), collapsed runs in the upper strip on a log scale. All models are trained with final loss; Table 6 lists the values. construction and next-token prediction separate residual streams and
- p.15 **Table 2**: Base model configurations. Looped models reuse the listed L-layer stack.
- p.16 **Table 3**: Seed variance at S. Final validation perplexity, mean ± 95% confidence interval (Student-t) over four seeds. In the STD row, R = 1 is the standard Transformer and R > 1 the deeper, narrower parameter-matched Transformer. SPS is the two-recursion baseline of Ap
- p.16 **Table 4**: Paired tests at S. Same-seed perplexity gap of the first model over the second, mean ± 95% confidence interval over four seeds, the gap in units of the standard deviation of the same-seed differences, and the one-sided paired t-test for the first model having 
- p.17 **Table 5**: LPT outperforms LT with the best normalization we tried. Validation perplexity (↓) after 20B tokens. Bold marks the best per column and underline the best LT. Values above 2,000 are unigram-level. A dash marks a run that did not finish. S
- p.18 **Figure 9**: Shared and local keys remain distinguishable with full prediction caches. Expanded comparison for Figure 6, including ω = 4,096. Relative depth
- p.18 **Figure 10**: Query/key geometry across LPT models. S and M models at three and five recursions, with final loss and ω = 64. Columns match relative layer depth. hidden-state path contribution, ghidden . Subtracting it from the ordinary first-recursion gradient g (1) gives t
- p.19 **Figure 11**: Attention to shared memory across training objectives and window sizes. Expanded comparison for Figure 5. Mean rows average all layers.
- p.19 **Figure 12**: Shared-cache feedback persists across training objectives and window sizes. Expanded version of Figure 7, with the same sequences and final-loss gradient probe in every condition. measurements, without clipping. Whiskers are 95% sequence-bootstrap intervals fo
- p.20 **Figure 13**: Shared-KV feedback relative to the hidden-state path. Gradient-norm ratios at fresh initialization and after training. Whiskers: 95% sequence-bootstrap intervals. Gradient Paths at Initialization Figure 13 compares the norm ratio ∥gKV ∥2 /∥ghidden ∥2 before an
- p.20 **Table 6**: Detaching shared-cache gradients hurts learning. Validation perplexity (↓) at S and M, all models trained with final loss; values above 1,000 are runs that failed to train. Bold marks the best per column. S
- p.21 **Table 7**: Quality and efficiency through nine recursions. One table per size. Metrics and formatting follow Table 1: in the STD row, R = 1 is the standard Transformer and R > 1 is DEEPSTD. SPS appears in the R = 2 column, whose memory and FLOPs it matches. S
- p.22 **Figure 14**: Quality and memory through nine recursions. Validation perplexity, mean five-task accuracy, and context memory relative to STD. Intermediate Readouts We apply the model’s readout normalization and language-modeling head to the output of each recursion. The mea
- p.23 **Figure 15**: Quality versus context memory across model sizes. Curves vary recursion count. Marker size denotes recursions, and shade denotes model size. S/M train for 20B tokens and L for 50B.
- p.23 **Figure 16**: Quality and memory across model sizes. Mean five-task accuracy and context memory for STD and looped models at nine recursions.
- p.25 **Table 8**: Decoding capacity and speed on one H100. One table per size. bf16, 4,096-token context. Batch: the largest batch that decodes without running out of the 80 GB. Tok/s: generated tokens per second at that batch. Lat.: milliseconds per decoding step at batch 1. O
- p.26 **Figure 17**: H - LPT is better than STD at similar or higher throughput. Validation perplexity (top) and mean five-task accuracy (bottom) against generated tokens per second at each model’s maximum batch, relative to same-size STD, on one H100 at a 4,096-token context. Cur

## Numbered equations seen (equation number, page)

- (none detected)

## Rendered pages (view with Read)

- reports/2610.02383_extract/pages/page-01.png
- reports/2610.02383_extract/pages/page-05.png
- reports/2610.02383_extract/pages/page-07.png
- reports/2610.02383_extract/pages/page-08.png
- reports/2610.02383_extract/pages/page-09.png
- reports/2610.02383_extract/pages/page-18.png
- reports/2610.02383_extract/pages/page-19.png
- reports/2610.02383_extract/pages/page-20.png
- reports/2610.02383_extract/pages/page-22.png
- reports/2610.02383_extract/pages/page-23.png
