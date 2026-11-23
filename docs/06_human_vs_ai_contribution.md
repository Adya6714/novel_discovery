# Human versus AI contribution

| Element | Human (repository owner) | AI agent (Claude, Anthropic) |
|---|---|---|
| Mission statement, 13 seed programmes, output requirements | Wrote them | — |
| Choice of repository | Provided URL | — |
| Network/environment change for literature access | Approved the prompt | Proposed it after arXiv was blocked |
| Landscape audit, ranking, selection of directions | — | All |
| Literature searches, reading of full texts, novelty matrix | — | All |
| Hypotheses, preregistration, experimental design | — | All |
| Code (training, interventions, attention analysis, lonely-runner C kernel, exact ML) | — | All |
| Running experiments, analysis, figures | — | All |
| Bug detection and fix (target off-by-one, D10) | — | Agent, via its own preregistered gate |
| Adversarial review | — | A separate AI agent instance with no access to the hypothesis-construction reasoning beyond the written documents. **This is not independent human peer review and not experimental replication.** |
| Final judgment on publication | Pending (human) | Self-assessment only, `docs/07_publication_readiness.md` |

No physical experiments were performed. All "experiments" are computational and are fully re-runnable from the repository.
