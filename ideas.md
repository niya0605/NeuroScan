# NeuroLens design direction

## Approach 1

**Theme Name:** Clinical Signal

**Very Brief Intro:** A restrained research-interface direction using warm paper tones, precise blue-green accents, and evidence-first information hierarchy. It feels like a high-quality medical visualization notebook rather than a generic dashboard.

**Probability:** 0.07

## Approach 2

**Theme Name:** Midnight Radiology

**Very Brief Intro:** A dark, cinematic diagnostic workspace with luminous heatmap accents and a focused interpretation console. The mood is advanced, technical, and suited to deep model inspection.

**Probability:** 0.04

## Approach 3

**Theme Name:** Atlas of Evidence

**Very Brief Intro:** An editorial atlas aesthetic with off-white panels, cartographic linework, and large typographic anchors. It frames model outputs as a guided visual story for researchers and reviewers.

**Probability:** 0.09

## Chosen Approach: Clinical Signal

**Design Movement:** Swiss International Typographic Style translated into a contemporary clinical research instrument.

**Core Principles:** Evidence before decoration; asymmetric editorial composition; quiet confidence through generous whitespace; every interaction should clarify model behavior rather than add novelty.

**Color Philosophy:** Warm ivory backgrounds keep the interface calm and legible, while deep ink provides scientific authority. The signature sea-glass teal marks model activity and explainability, and a restrained coral note signals uncertainty or attention without making diagnostic claims.

**Layout Paradigm:** A left rail establishes context while the main canvas moves from headline insight to visual evidence to probability detail. The upload surface is a horizontal instrument panel, not a centered hero card; results unfold as stacked research plates with a persistent model-status spine.

**Signature Elements:** Thin measurement rules with small index labels; teal diagnostic crosshair motif; rounded image windows that feel like viewports into a scan rather than generic cards.

**Interaction Philosophy:** Interactions should expose a layer of reasoning. Hovering or selecting a class updates explanatory copy and emphasis; toggles reveal original versus attention overlay; upload actions feel like placing a specimen into a controlled analysis tray.

**Animation:** Use short 160–240ms ease-out transitions for panel emphasis, probability bar growth, tab changes, and image-layer crossfades. Stagger only major result plates by 40ms. Respect reduced motion and avoid continuous decorative animation.

**Typography System:** Use DM Sans for interface labels and controls, paired with IBM Plex Serif for editorial headlines and callouts. Headline hierarchy is compact and left-aligned; uppercase micro-labels use generous tracking; numbers use tabular figures for easy comparison.

**Brand Essence:** NeuroLens is an explainable MRI classification showcase for researchers and reviewers who want to see not only what the model predicted, but where it looked.

Personality: rigorous, lucid, measured.

**Brand Voice:** Headlines are direct and observational. CTAs sound like lab actions, not marketing promises. Microcopy names uncertainty clearly and avoids clinical overreach.

Example lines: “See where the model looked.” “Place an MRI into the analysis tray.”

**Wordmark & Logo:** A compact four-quadrant lens mark made from a rounded crosshair and one offset teal quadrant, paired with a custom-spaced NEUROLENS wordmark. The symbol should remain legible at favicon size without relying on text.

**Signature Brand Color:** Sea-glass teal `#0F8B8D`, used sparingly for model-state emphasis and explainability controls.

## Style Decisions

The showcase will prioritize transparent, non-diagnostic language: it presents model outputs as research artifacts and labels the supplied model’s four classes as `glioma`, `meningioma`, `notumor`, and `pituitary`. The frontend will provide a representative interactive analysis state because the static project cannot run TensorFlow inference in-browser; the supplied `app.py` remains the reference for the intended upload, prediction, probability, and Grad-CAM flow.
