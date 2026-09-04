# Roadmap

v0.1.0 (this release) integrates one provider (LIBERO) end to end with one reference policy
(Behavior Prompting Policy) on one backend (vla-eval). The standard is written for many of each.

## Providers (tracks they unlock)

| provider | backend | tracks | notes |
|---|---|---|---|
| RoboCasa365 | vla-eval image | instance, category, scene, composition, long_horizon | MIT/CC-BY; per-frame subtask labels; mobile base → `not_applicable` for fixed-base policies |
| RoboTwin 2.0 | vla-eval image / XPolicyLab | embodiment (same tasks on 5 robots), scene, instance | bimanual joint actions |
| ManiSkill3 | vla-eval / native | dynamics (physics randomisation), embodiment, configuration | CC-BY-NC assets |
| CALVIN | vla-eval image | long_horizon, scene | 166 GB+ |
| MetaWorld ML45 | lerobot-eval / native | unseen_skill, configuration | state-based sanity track |
| LIBERO-Gen / LIBERO-PRO / LIBERO-Plus | vla-eval | composition (Gen chains), perturbations | additional task sets under the `libero` provider |
| DynaMimicGen | native robosuite | dynamics (change during rollout) | generator, not a suite |
| RLBench (+AGNOSTOS, GemBench) | vla-eval (licence-gated) | unseen_skill, composition | non-commercial licence: optional, never redistributed |
| RH20T / MIME | context only | prompt modality `human_video` | demonstration providers without rollouts |

## Tracks and conditions

`instance`, `category`, `composition`, `unseen_skill`, `embodiment`, `dynamics`, `long_horizon`
(declared, pools empty until a provider contributes); controls `distractor`, `reversed_actions`,
`conflict` (wrong demonstrations + correct instruction), `language_oracle`,
`expert_upper_bound`; prompt modalities `robot_video`, `human_video`.

## Policies

ICRT (Apache-2.0, LIBERO-trainable via BPP's baseline), RoboSSM (LIBERO-native, needs training),
Instant Policy (point clouds; cross-embodiment), RICL (retrieval, DROID), language-conditioned
and goal-image BPP checkpoints as non-ICIL reference rows.

## Backends and bridges

XPolicyLab policy bridge (`ModelTemplate`, context via `prepare_case`) and its RoboDojo/RoboTwin
environments; lerobot-eval backend; cross-session batched inference in the model server; upstream
vla-eval proposals (a `resolution` kwarg for LIBERO, a first-class `context` field at
`EPISODE_START`).

## Data and publishing

Hub publishing of the converted LeRobotDataset v3 demonstration sets with the S2 dataset card;
a training-side K-shot sampler that reuses the evaluation sampler; results publishing and a
leaderboard site reading `icil_results.json`.
