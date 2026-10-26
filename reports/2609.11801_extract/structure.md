# Structure of 2609.11801.pdf

- Pages: 19; extracted characters: 60400
- References start on page 10 (body is pages 1-10; later pages are references/appendix)

## First lines of page 1 (title / authors / affiliations)

    Thinking with Looped Flows
    Ayhan Suleymanzade1,∗ , Chanhyuk Lee2 , Floor Eijkelboom3
    Nicholas M. Boffi4 , İsmail İlkan Ceylan5,6,7,† , Jinwoo Kim2,†
    arXiv:2609.11801v1 [cs.LG] 10 Sep 2026
    1
    EPFL
    2
    KAIST 3 University of Amsterdam 4 Carnegie Mellon University
    5
    TU Wien 6 AITHYRA 7 University of Oxford

## Abstract (auto-detected, verify against paper.txt)

Humans and machines often solve harder problems by spending more time on computation. In deep learning, looped models implement this idea during inference by recurrently updating a hidden state. In practice, however, their training backpropagates through only one or a few updates, making it hard to train early updates to support future ones. We propose looped flows, an approach that sidesteps this issue by training the recurrence with local denoising objectives. By imposing temporal association across denoising objectives through progressively decreasing noise levels and shared noise, the model is incentivized to learn recurrent states that transfer useful computation over time, even when gradients cover only a few updates. We then formulate inference as integrating the velocity of a probability flow parameterized by the learned denoiser, coupled with recurrent states. This allows solving harder problems by spending more computation through a finer temporal grid and enables multiple valid predictions from different initial noise samples. Across six reasoning benchmarks including two multi-solution benchmarks, looped flows outperform prior state-of-the-art looped models overall, achieving 58.8% test accuracy on ARC-AGI-1 and 12.2% on ARC-AGI-2. Recurrent state zti Dti Denoising xti Problem c ... Noise x0 ... ODE/SDE step Flow state xti Updated state xti+1 Solution x1 Figure 1 Thinking with looped flows. At each step, a stateful denoiser predicts a solution and updates the recurrent state, and then an ODE/SDE step updates the flow state. This enables performance to improve with additional inference-time computation and supports diverse predictions through probability transport.

## Headings (page: heading)

- p.1: Introduction
- p.3: Background
- p.6: Related Work
- p.6: Experiments
- p.7: Method
- p.10: Conclusion
- p.10: Acknowledgments
- p.10: References
- p.14: Appendix
- p.14: Architecture

## Figure and table captions

- (none detected)

## Numbered equations seen (equation number, page)

- Eq. (14) p.10
