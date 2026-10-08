# Prior art and reviewer baselines for N1, N2, N3, C1, C3

(Saved by the coordinator from the researcher's returned text; its file write was blocked.)
Tags: [VERIFIED] = page/abstract/code/table opened; [SNIPPET] = search snippet only; [INFERENCE] = reasoning. Research date 2026-10-01.

## Shared context
- No reproduction or critique of YOLIC exists. Semantic Scholar lists 11 citing works of arXiv:2307.06689, none critical; several are by the YOLIC group. [VERIFIED] https://api.semanticscholar.org/graph/v1/paper/arXiv:2307.06689/citations?fields=title,year,venue,externalIds&limit=100
  - YOLIC labeling tool with SAM (SoftwareX 2026); "Efficient and Fault-tolerant ... Ensemble of Dual Ternary YOLIC Models" (MCSoC 2024); "Cost-Sensitive Road Obstacle Detection for Low-Cost Autonomous Vehicles" (MobileCloud 2024).
- Cost-sensitive YOLIC: Su, Zhao, Tomioka, Liu, MobileCloud 2024, DOI 10.1109/MobileCloud62079.2024.00011. Builds on YOLIC by "assigning higher weights to proximate vehicle areas" (distance, direction, driving relevance); real-time on Pi 4B. [VERIFIED abstract via S2 API] https://api.semanticscholar.org/graph/v1/paper/search?query=Cost-Sensitive%20Road%20Obstacle%20Detection%20for%20Low-Cost%20Autonomous%20Vehicles&fields=title,year,venue,authors,abstract,externalIds&limit=3 . Full text not opened (https://ieeexplore.ieee.org/iel8/10685161/10685162/10685441.pdf); unknown whether weights are in the loss or only the metric.
- Ternary ensemble: three ternary YOLIC models, "comparable accuracy", 89.5% (79.7%) compute reduction vs TMR. [SNIPPET] https://ieeexplore.ieee.org/document/10819538/
- "A Selective Multi-Branch Network for Edge-Oriented Object Localization and Classification", Electronics 13(8):1472, 2024 — 403, not verified. https://www.mdpi.com/2079-9292/13/8/1472

## N1 — GAP+FC absolute-position encoding
### Takeaway
Already established that zero padding makes CNNs encode absolute position (ICLR 2020 → IJCV 2024; CVPR 2020). Most damaging: Islam et al. ICCV 2021 show position information survives global pooling, encoded channel-wise — the YOLIC mechanism question answered in general form. N1 would confirm known results; only an application-specific quantitative design rule might be new. Reviewer's demanded baseline = spatial head (E1), likely reducing N1 to a motivating section. UFLD does not use GAP.
### Cited findings
- Islam et al., "Position, Padding and Predictions", IJCV 2024: "a surprising degree of absolute position information is encoded in commonly used CNNs ... zero padding drives CNNs to encode position information ..., while a lack of padding precludes position encoding"; quantify boundary effects "as a function of the distance to the border"; position info "can both help or hurt performance". [VERIFIED abstract] https://arxiv.org/abs/2101.12322 ; https://link.springer.com/article/10.1007/s11263-024-02069-9
- Islam et al., "Global Pooling, More than Meets the Eye: Position Information is Encoded Channel-Wise in CNNs", ICCV 2021: "positional information is encoded based on the ordering of the channel dimensions, while semantic information is largely not". [VERIFIED abstract] https://arxiv.org/abs/2108.07884 ; https://openaccess.thecvf.com/content/ICCV2021/papers/Islam_Global_Pooling_More_Than_Meets_the_Eye_Position_Information_Is_ICCV_2021_paper.pdf
- Kayhan & van Gemert, CVPR 2020: conv filters exploit absolute location via boundary effects, "even far from the image boundary". [SNIPPET] https://jvgemert.github.io/pub/kayhanCVPR20translationInvarianceCNN.pdf
- Alsallakh et al., "Mind the Pad", ICLR 2021: padding causes spatial bias / blind spots hurting small-object detection. [SNIPPET] https://openreview.net/forum?id=m1CD7tPubNy
- Lin et al., arXiv 2206.01202 (2022): better position-information metrics than PosENet; padding-induced pattern "is primarily a learning artifact and is less dependent on the characteristics of the underlying padding schemes". [VERIFIED abstract] https://arxiv.org/abs/2206.01202 → [INFERENCE] padding-swap probes may show weaker effects than expected; reviewers expect these metrics.
- Garcia-Gasulla et al., "Padding Aware Neurons", ICCV-W 2023: border-detecting filters, "dozens to hundreds per network". [VERIFIED abstract] https://arxiv.org/abs/2309.08048
- Others [SNIPPET]: Random Padding (arXiv 2302.08682), PadChannel (https://arxiv.org/abs/2311.07623), Murase (arXiv 2005.03463).
- UFLD (Qin et al., ECCV 2020): "row-based selecting ... using global features" [VERIFIED] https://arxiv.org/pdf/2004.11757 . Head: Conv2d(512,8,1) → view(-1,1800) → Linear(1800,2048) → ReLU → Linear(2048,total_dim) [VERIFIED code] https://raw.githubusercontent.com/cfzd/Ultra-Fast-Lane-Detection/master/model/model.py — flattens the spatial map; no GAP.
### Inferences
- N1's headline is a special case of Islam ICCV 2021 + IJCV 2024.
- Confound: fixed camera → content correlates with position (perspective scale, road vs sky); GAP features could locate cells without padding cues. Translation/crop probes can't separate these; changing padding at train time breaks ImageNet weights.
- Only possibly publishable angle: per-cell error vs cell size and border distance on a real fixed-camera task across 2 backbones → design rule. If E1 matches/beats GAP+FC everywhere (likely), the rule is "don't use GAP" and N1 becomes a section of an E1 paper.
- Feasibility high (Cityscapes relabelable); per-cell differences need ≥3 seeds.
### Gaps
Full texts of ICLR 2020, IJCV 2024, Kayhan not opened. No 2024–2026 analysis of global-FC structured heads found (limited search).

## N2 — Re-evaluation of YOLIC
### Takeaway
No YOLIC reproduction exists. Reproductions are published mostly in ReScience C and MLRC (→ TMLR). Single-paper re-evaluation is a weak fit for an IEEE journal; IEEE Access plausible only if framed as benchmark/protocol with stronger baselines and multi-seed statistics. Likely findings (leakage, broken baselines) probably exceed seed noise.
### Cited findings
- 11 citing works, all non-critical [VERIFIED] (S2 link above): SoftwareX 2026; J. Ambient Intell. 2026; SN Comp. Sci. 2026 (MLRDv2); Sci. Rep. 2025; arXiv TinyEcoWeedNet 2025; Neurocomputing 2025; arXiv micromobility review 2025; ETRI J 2025; MCSoC 2024; Appl. Sci. 2024; MobileCloud 2024.
- ReScience C publishes replications with open review; not IEEE. [SNIPPET] https://rescience.github.io/
- MLRC: reports to TMLR via OpenReview; MLRC 2026 an official NeurIPS 2026 track. [VERIFIED] https://reproml.org/
- Asselin et al., arXiv 2405.06911: DETR and ViTDet "could not achieve accuracy or speed performances comparable to what is declared"; RTMDet and YOLOv7 "could match". [VERIFIED abstract] https://arxiv.org/abs/2405.06911 . Another CV replication under review at ReScience C [SNIPPET] https://arxiv.org/pdf/2406.03586
- IEEE Access: Alahmari et al. 2020, "Challenges for the Repeatability of Deep Learning Models". [SNIPPET] https://ecommons.luc.edu/cgi/viewcontent.cgi?article=1410&context=cs_facpubs
- Barz & Denzler (ciFAIR): 3.3% / 10% near-duplicate test images in CIFAR-10/100; re-evaluation-only paper published in J. Imaging. [SNIPPET] https://arxiv.org/abs/1902.00423
- Bouthillier et al., MLSys 2021: data split is the largest variance source; init < 50% of bootstrap variance. [SNIPPET] https://arxiv.org/pdf/2103.03098 ; also arXiv 2109.08203.
### Inferences
- Best IEEE target: IEEE Access as "leakage-free multi-seed benchmark for cell-wise detection with corrected baselines", stronger bundled with a method (E1/C2). Standalone → ReScience C / MLRC.
- Baselines reviewers demand: (1) sequence-aware vs random split delta for same model; (2) light segmentation pooled to cells; (3) YOLO on real Cityscapes instance boxes rasterised to cells.
- Data risk: outdoor/indoor have only cell labels, apparently no sequence IDs → real-box YOLO only on Cityscapes; pHash-clustered splits will be questioned. Cityscapes is already city-disjoint, so leakage applies only to in-house sets.
- Cityscapes margins < 1 F1 (0.8202 vs 0.8134) may become non-significant with seeds.
- Contact authors about D1–D3 before claiming errors.

## N3 — Accuracy vs cell size / count
### Takeaway
Stride/resolution ablations are standard (DeepLabv3 output-stride table; IJCV 2024 border distance). A YOLIC cell-size curve is cheap but reads as YOLIC's missing ablation unless it yields a predictive rule across backbones/resolutions and disentangles the confound that smaller cells lower positive rate and add label noise.
### Cited findings
- DeepLabv3 Table 1 (VOC val) mIoU by output stride 8/16/32/64/128/256: 75.18/73.88/70.06/59.99/42.34/20.29. [VERIFIED] https://ar5iv.labs.arxiv.org/html/1706.05587
- Resolution-aware atrous rates, arXiv 2307.14179 [SNIPPET]; small-object segmentation size sensitivity, arXiv 2309.14117 [SNIPPET].
### Inferences
- No scaling-law study for fixed-layout cell classification found (limited search; occupancy-grid literature not covered).
- Under ≥1-pixel rule, smaller cells → fewer positives, more sliver noise → must use AP / prior-normalised metrics.
- Interesting result = GAP+FC collapses earlier than spatial head/seg-pooled → merges N3 with N1/E1.
- Compute: 10–20 layouts × 2 backbones × 2–3 resolutions × 3 seeds = 120–360 runs at 3–5 h → over budget on one RTX 4050 unless shortened.

## C1 — Imbalance losses
### Takeaway
ASL ≈ +2.6 mAP over CE on COCO (TResNet-L@448), single runs. For independent sigmoids, post-hoc logit adjustment ≡ per-label threshold tuning, so BCE + tuned thresholds is the required F1 control; loss changes must win on threshold-free AP over ≥3 seeds. YOLIC group already has cost-sensitive YOLIC (2024). Low novelty; ablation table at best.
### Cited findings
- ASL (Ridnik et al., ICCV 2021): COCO TResNet-L@448 CE 84.0 / focal 85.1 / ASL 86.6 mAP; VOC ASL 94.6 (ImageNet pretrain), 95.8 (COCO pretrain); ML-GCN 94.0, SSGRL 93.4. [VERIFIED] https://ar5iv.labs.arxiv.org/html/2009.14119 ; https://openaccess.thecvf.com/content/ICCV2021/papers/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.pdf . No seeds/variance reported (per fetch tool, likely).
- ASL reduces to BCE at γ=0, m=0 [SNIPPET] https://arxiv.org/pdf/2304.05361
- Logit adjustment (Menon et al., ICLR 2021) [SNIPPET] https://iclr.cc/virtual/2021/poster/2675
- Two-way loss, CVPR 2023 [SNIPPET] https://openaccess.thecvf.com/content/CVPR2023/papers/Kobayashi_Two-Way_Multi-Label_Loss_CVPR_2023_paper.pdf ; Asymmetric Polynomial Loss, ICASSP 2023 [SNIPPET] https://ieeexplore.ieee.org/document/10095437/
### Inferences
- [math] σ(z−τ) > 0.5 ⇔ z > τ, so post-hoc prior correction gains on F1 are matched by validation threshold tuning.
- Baselines: BCE + per-class (or class×cell-group) tuned thresholds; BCE with pos_weight; focal; ≥3 seeds.
- Gain < 1 mAP likely within noise; rare-class gains (TrafficSign, People) might survive with per-class CIs.
- Sliver positives under ≥1-pixel rule are label noise; focal up-weighting may amplify it.

## C3 — CRF / GNN over cell graph
### Takeaway
Structured modules give small gains over strong CNNs (DeepLabv3 dropped DenseCRF; ASL without label-correlation beat ML-GCN on VOC). On 100–250 cells where every output sees global features, a logit-refinement MLP or spatial conv head will probably match, within seed noise.
### Cited findings
- DeepLabv3: improves "without DenseCRF post-processing", 85.7% VOC test. [VERIFIED] https://arxiv.org/abs/1706.05587
- ASL 94.6 vs ML-GCN 94.0 VOC (not controlled). [VERIFIED] https://ar5iv.labs.arxiv.org/html/2009.14119
- CRF-as-RNN, ICCV 2015 (tables not retrieved) https://arxiv.org/abs/1502.03240 ; graph reasoning in segmentation, arXiv 2108.03791 [SNIPPET]
### Inferences
- All YOLIC outputs are linear in one shared pooled vector → co-occurrence/extent structure partly captured already.
- Baselines: 2-layer MLP on N×(M+1) logits (more general than grid CRF with tied compatibilities); post-hoc 3×3 majority/morphological filter; E1 spatial head.
- Self-defined "fragmentation" metrics will be discounted; F1/AP gains likely < 1 point; ≥5 seeds + paired tests needed.
- Temporal (A3) structure more defensible than spatial CRF for this application.

## Summary verdicts [INFERENCE]
| Idea | Already done? | Verdict |
|---|---|---|
| N1 | Largely (Islam ICCV 2021; IJCV 2024) | Weak standalone; section of E1. UFLD flattens, not GAP. |
| N2 | No YOLIC reproduction | Effects likely exceed noise; IEEE Access as benchmark bundled with a method, or ReScience C / MLRC. |
| N3 | Ablation-level | Confounded by positive-rate / label noise; compute-heavy. |
| C1 | Low novelty; partial overlap with cost-sensitive YOLIC | Tuned-threshold BCE likely dominates F1; must show AP gains with seeds. |
| C3 | — | Likely dominated by logit-MLP / spatial head / smoothing; within noise. |
