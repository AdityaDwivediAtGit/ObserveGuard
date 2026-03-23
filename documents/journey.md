# My Journey Behind the ObserveGuard Paper: From Industry Firefighting to IEEE Submission

**Author**: Aditya Dwivedi
**Role**: Practicing Data & AI Engineer (1.5+ years industry experience)  
**Location**: Delhi, India  
**Date**: March 2026  
**Paper**: "ObserveGuard: Observation-Centric Secure Multimodal Agents for Trustworthy Edge Deployment"  
**Target**: IEEE AI-themed competition TACTiCS conference submission

## 1. The Spark: Daily Pain in Production (Early 2025–Late 2025)

As a Data & AI Engineer in Delhi-based teams (fintech, IoT, retail analytics), most of my days were spent firefighting production issues rather than building new models.

- Multimodal agents (vision + audio + text) on edge devices (Raspberry Pi clusters, Jetson boards) for real-time IoT tasks kept failing silently.
- UI mutations in Android dashboards, sensor noise in factory cameras, and subtle cross-app overlays caused "observation-to-action gaps" — agents executed wrong actions with no clear logs.
- These were real rebinding-like exploits (later confirmed in emerging 2025–2026 literature) that bypassed permissions and triggered costly incidents.
- Weekly pattern: incident tickets, manual rollbacks, emergency retraining, frustrated stakeholders, and rising cloud-sync costs.

I kept wondering:  
"Why do we trust raw observations so blindly in production agent loops?"  
"Why isn't there a lightweight, on-device guard that enforces atomic consistency before reasoning?"

No existing tool (LangChain observability, Evidently drift monitors, or commercial MLOps) fully addressed this for multimodal edge agents.

## 2. The Turning Point: Deciding to Build It (December 2025 – January 2026)

In early December 2025, after yet another weekend on-call fixing edge agent failures, I decided to turn the pain into a proper solution.

- Started reading recent papers: OSWorld updates, AppAgent extensions, and the emerging 2026 preprints on GUI agent vulnerabilities.
- Reproduced OSWorld benchmark tasks locally over weekends (thanks to the open xlang-ai repo).
- First quick prototype (mid-December 2025): basic cosine-similarity check between observations → detected some issues but flooded with false positives and hurt throughput.
- Realized I needed more: predictive modeling, synthetic probes, formal bounds, and energy/carbon awareness to make it TACTiCS-competitive.

## 3. Intensive Development & Breakthrough (January – March 2026)

Over the next three intense months (working 10–20 hours/week after my day job):

- **January 2026**  
  - Formulated the "observe-first-then-decide" protocol.  
  - Built the lightweight transition model (MLP trained contrastively on clean trajectories).  
  - Added synthetic probe injection (rule-based + lightweight LoRA priors for UI/audio/text).  
  - Ran initial tests on Raspberry Pi 5 → promising rebinding detection but needed majority-vote verification.

- **February 2026**  
  - Derived the exponential rollback probability bound (P(undetected) ≤ e^{-κK}).  
  - Conducted full ablations: probe count (K=3 optimal), threshold τ sensitivity, noise levels (20–40%).  
  - Integrated CodeCarbon for energy profiling → achieved 22% reduction under 5W cap.  
  - Tested on drifted OSWorld + SSv2 extensions: +13% success rate, 100% rebinding mitigation vs. baselines.

- **Early March 2026**  
  - Ran a short A/B test in a live industrial IoT pipeline (vision-audio quality inspection agent).  
  - Result: 64% drop in monthly incident volume over the test period, with <4 ms added latency and 2.1× throughput under power constraints.  
  - Addressed ethics: debiased probe generation, on-device privacy (no cloud leaks), carbon footprint reduction.

## 4. Paper Writing & Final Polish (March 2026)

- Mid-March 2026: Drafted the full IEEE double-column LaTeX paper (10 pages max).  
- Expanded Related Work with critical analysis of 18+ recent papers (2023–2026).  
- Added TikZ diagrams (architecture + protocol flowchart) for visual clarity.  
- Built complete reproducibility package: experiments.md, Docker setup, fixed seed=42, public datasets only.  
- Wrote mandatory sections: ethics/bias/privacy/carbon impact, deployment scalability.  
- Final review: confident tone, every claim backed by experiment or citation, compelling abstract/intro.

Submitted to the AI Theme Competition 2026 TACTiCS on March 23, 2026.

## 5. What I Learned & Why It Matters

This was not a multi-year PhD project — it was a focused 3-month sprint by an industry engineer solving a real production problem.

Key takeaways:
- Real incidents are the best source of novelty — production pain points often precede academic gaps.
- IEEE prioritizes deployability, sustainability, and trustworthiness — align tightly with those.
- Reproducibility is king: open code, fixed seeds, edge hardware logs win judges.
- Possible to balance full-time work + research with discipline (evenings/weekends).
- Tangible impact: the guard layer is already reducing risk in one live pipeline.

If accepted, this proves industry practitioners can contribute meaningfully to trustworthy edge AI at scale.

Future ideas: federated guard updates, neurosymbolic probes, multi-agent coordination guards.

Grateful for open-source community (OSWorld, Hugging Face, CodeCarbon), IEEE resources, and late-night debugging sessions.

Aditya Dwivedi
Delhi, March 2026
