# Report Alignment with Coursework Requirements

## Research question

Does a multi-agent debate architecture improve predictive performance, robustness and explanation coverage for six-month US-target conflict escalation forecasting compared with model-only and single-agent baselines?

## Environment

The environment is a temporal geopolitical event environment. Each state is a US-target dyad-month constructed from ACLED event data. Agent actions take the form of risk assessments that affect the final Judge Agent probability and the generated risk report.

## Agents

- Sentiment Agent
- Escalation Agent
- De-escalation Agent
- Geo-context Agent
- Data-driven Agent
- Judge Agent

## Experiments

1. Architecture comparison: Logistic Regression, Random Forest, LSTM, Single-Agent RAG-style, Multi-Agent Debate.
2. Multi-head evaluation: compare individual agent heads against the full Judge system.
3. Noise robustness: perturb features and measure degradation using mean and standard deviation.
4. Current prediction demonstration: estimate six-month risk using the latest six months of ACLED data.

## Added value

The added value is not just downloading ACLED or training a classifier. The project transforms event data into an agent-based experimental environment, designs specialised autonomous reasoning agents, compares several architectures, and evaluates robustness under noise.

## Safe wording

Use:

> high conflict escalation involving the United States and a target state

Avoid:

> the United States will start a war

## Limitations

- ACLED labels represent event escalation, not formal declarations of war.
- Some target countries may be missing from the downloaded ACLED subset.
- Current predictions cannot be validated until the horizon has passed.
- No policy or military recommendation is generated.
