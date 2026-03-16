"""
Fuzzy Decision Engine for DDS-LLM Orchestrator.

Uses scikit-fuzzy to select agents, QoS profiles, and execution strategies
based on task urgency, complexity, agent load, and agent latency.

The engine builds a fuzzy control system once at construction time (~50ms)
and runs inference per-request (~0.3ms per agent evaluation).
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    logger.warning("scikit-fuzzy not installed. Run: pip install scikit-fuzzy")


@dataclass
class FuzzyInput:
    urgency: float        # 1-10 (from request priority or explicit field)
    complexity: float     # 1-10 (estimated from prompt size)
    agent_load: float     # 0-100 (% slots used)
    agent_latency: float  # 0-2000 (ms, historical average)


@dataclass
class FuzzyOutput:
    agent_score: float    # 0-100 (higher = better candidate)
    qos_score: float      # 0-10 (mapped to QoSProfile via profile_from_score)
    strategy_score: float # 0-10 (mapped to strategy string)


@dataclass
class FuzzyDecision:
    agent_id: str
    agent_score: float
    qos_profile: str       # "low_cost" | "balanced" | "critical"
    strategy: str          # "single" | "retry" | "fanout"
    all_scores: Dict[str, float] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)
    inference_time_ms: float = 0.0


class FuzzyDecisionEngine:
    """Fuzzy inference engine for orchestrator decisions.

    Evaluates each candidate agent independently and returns the best one
    along with QoS profile and execution strategy.
    """

    def __init__(self):
        if not FUZZY_AVAILABLE:
            raise ImportError("scikit-fuzzy is required. Install: pip install scikit-fuzzy")

        self._ctrl_system = None
        self._build_fuzzy_system()
        logger.info("Fuzzy decision engine initialized (18 rules)")

    def _build_fuzzy_system(self):
        """Build the fuzzy control system with 4 inputs, 3 outputs, 18 rules."""

        # ─── Input variables (Antecedents) ───────────────────────────────
        urgency = ctrl.Antecedent(np.arange(0, 11, 0.5), 'urgency')
        complexity = ctrl.Antecedent(np.arange(0, 11, 0.5), 'complexity')
        agent_load = ctrl.Antecedent(np.arange(0, 101, 1), 'agent_load')
        agent_latency = ctrl.Antecedent(np.arange(0, 2001, 10), 'agent_latency')

        # ─── Output variables (Consequents) ──────────────────────────────
        agent_score = ctrl.Consequent(np.arange(0, 101, 1), 'agent_score')
        qos_profile = ctrl.Consequent(np.arange(0, 11, 0.5), 'qos_profile')
        strategy = ctrl.Consequent(np.arange(0, 11, 0.5), 'strategy')

        # ─── Membership functions ────────────────────────────────────────

        # Urgency: low/medium/high
        urgency['low'] = fuzz.trimf(urgency.universe, [0, 0, 4])
        urgency['medium'] = fuzz.trimf(urgency.universe, [3, 5, 7])
        urgency['high'] = fuzz.trimf(urgency.universe, [6, 10, 10])

        # Complexity: low/medium/high
        complexity['low'] = fuzz.trimf(complexity.universe, [0, 0, 4])
        complexity['medium'] = fuzz.trimf(complexity.universe, [3, 5, 7])
        complexity['high'] = fuzz.trimf(complexity.universe, [6, 10, 10])

        # Agent load: low/medium/high (0-100%)
        agent_load['low'] = fuzz.trimf(agent_load.universe, [0, 0, 30])
        agent_load['medium'] = fuzz.trimf(agent_load.universe, [20, 50, 80])
        agent_load['high'] = fuzz.trimf(agent_load.universe, [70, 100, 100])

        # Agent latency: low/medium/high (0-2000ms)
        agent_latency['low'] = fuzz.trimf(agent_latency.universe, [0, 0, 200])
        agent_latency['medium'] = fuzz.trimf(agent_latency.universe, [100, 500, 1000])
        agent_latency['high'] = fuzz.trimf(agent_latency.universe, [800, 2000, 2000])

        # Agent score: low/medium/high (0-100)
        agent_score['low'] = fuzz.trimf(agent_score.universe, [0, 0, 40])
        agent_score['medium'] = fuzz.trimf(agent_score.universe, [30, 50, 70])
        agent_score['high'] = fuzz.trimf(agent_score.universe, [60, 100, 100])

        # QoS profile: low_cost/balanced/critical (0-10)
        qos_profile['low_cost'] = fuzz.trimf(qos_profile.universe, [0, 0, 3])
        qos_profile['balanced'] = fuzz.trimf(qos_profile.universe, [2, 5, 8])
        qos_profile['critical'] = fuzz.trimf(qos_profile.universe, [7, 10, 10])

        # Strategy: single/retry/fanout (0-10)
        strategy['single'] = fuzz.trimf(strategy.universe, [0, 0, 3])
        strategy['retry'] = fuzz.trimf(strategy.universe, [2, 5, 8])
        strategy['fanout'] = fuzz.trimf(strategy.universe, [7, 10, 10])

        # ─── Rules ──────────────────────────────────────────────────────

        rules = []

        # Agent scoring rules (R01-R07)
        rules.append(ctrl.Rule(urgency['high'] & complexity['low'], agent_score['high']))          # R01: fast agent for urgent simple
        rules.append(ctrl.Rule(urgency['high'] & complexity['high'], agent_score['high']))         # R02: quality agent for urgent complex
        rules.append(ctrl.Rule(urgency['low'] & complexity['low'], agent_score['low']))            # R03: backup OK for non-urgent simple
        rules.append(ctrl.Rule(complexity['high'] & agent_load['low'], agent_score['high']))       # R04: prefer free agent for complex
        rules.append(ctrl.Rule(complexity['medium'], agent_score['medium']))                        # R05: medium tasks get medium score
        rules.append(ctrl.Rule(agent_load['high'], agent_score['low']))                            # R06: penalize loaded agents
        rules.append(ctrl.Rule(agent_latency['high'], agent_score['low']))                         # R07: penalize slow agents

        # QoS profile rules (R08-R13)
        rules.append(ctrl.Rule(urgency['low'], qos_profile['low_cost']))                           # R08
        rules.append(ctrl.Rule(urgency['medium'], qos_profile['balanced']))                        # R09
        rules.append(ctrl.Rule(urgency['high'], qos_profile['critical']))                          # R10
        rules.append(ctrl.Rule(agent_latency['high'], qos_profile['critical']))                    # R11
        rules.append(ctrl.Rule(agent_load['high'] & urgency['medium'], qos_profile['balanced']))   # R12
        rules.append(ctrl.Rule(complexity['high'], qos_profile['balanced']))                       # R13

        # Strategy rules (R14-R18)
        rules.append(ctrl.Rule(urgency['high'] & agent_latency['high'], strategy['fanout']))       # R14
        rules.append(ctrl.Rule(agent_load['high'] & urgency['high'], strategy['fanout']))          # R15
        rules.append(ctrl.Rule(urgency['low'], strategy['single']))                                # R16
        rules.append(ctrl.Rule(agent_latency['medium'] & urgency['medium'], strategy['retry']))    # R17
        rules.append(ctrl.Rule(complexity['high'] & urgency['high'], strategy['fanout']))          # R18

        self._ctrl_system = ctrl.ControlSystem(rules)

    def evaluate(self, inp: FuzzyInput) -> FuzzyOutput:
        """Run fuzzy inference for a single agent evaluation.

        Creates a fresh simulation from the shared control system (thread-safe).
        """
        sim = ctrl.ControlSystemSimulation(self._ctrl_system)

        # Clamp inputs to valid ranges
        sim.input['urgency'] = max(0.0, min(10.0, inp.urgency))
        sim.input['complexity'] = max(0.0, min(10.0, inp.complexity))
        sim.input['agent_load'] = max(0.0, min(100.0, inp.agent_load))
        sim.input['agent_latency'] = max(0.0, min(2000.0, inp.agent_latency))

        try:
            sim.compute()
            return FuzzyOutput(
                agent_score=sim.output.get('agent_score', 50.0),
                qos_score=sim.output.get('qos_profile', 5.0),
                strategy_score=sim.output.get('strategy', 3.0),
            )
        except Exception as e:
            logger.warning(f"Fuzzy inference failed: {e}, using defaults")
            return FuzzyOutput(agent_score=50.0, qos_score=5.0, strategy_score=3.0)

    def select(self, task_input: dict, agents: list) -> FuzzyDecision:
        """Select the best agent using fuzzy inference.

        Args:
            task_input: dict with keys: urgency, complexity, messages, priority
            agents: list of AgentInfo objects

        Returns:
            FuzzyDecision with selected agent, QoS profile, strategy
        """
        t_start = time.perf_counter_ns()

        if not agents:
            return FuzzyDecision(
                agent_id="", agent_score=0.0, qos_profile="balanced",
                strategy="single", inference_time_ms=0.0,
            )

        # Extract task-level inputs
        urgency = float(task_input.get("urgency", 5))
        complexity = task_input.get("complexity")
        if complexity is None:
            complexity = self.estimate_complexity(task_input.get("messages", []))
        complexity = float(complexity)

        # Evaluate each agent
        best_agent = None
        best_score = -1.0
        best_output = None
        all_scores = {}

        for agent in agents:
            load_pct = 0.0
            slots_total = getattr(agent, "slots_total", 1) or 1
            slots_idle = getattr(agent, "slots_idle", 1)
            load_pct = max(0.0, (1.0 - slots_idle / slots_total) * 100.0)

            latency = getattr(agent, "avg_latency_ms", 0.0) or 0.0

            # Agent profile bonus: boost score for matching profiles
            profile = getattr(agent, "agent_profile", "balanced")
            profile_bonus = self._profile_bonus(profile, urgency, complexity)

            inp = FuzzyInput(
                urgency=urgency,
                complexity=complexity,
                agent_load=load_pct,
                agent_latency=latency,
            )

            output = self.evaluate(inp)
            final_score = output.agent_score + profile_bonus

            agent_id = getattr(agent, "agent_id", str(agent))
            all_scores[agent_id] = round(final_score, 2)

            if final_score > best_score:
                best_score = final_score
                best_agent = agent
                best_output = output

        t_end = time.perf_counter_ns()

        # Map output scores to discrete values
        from qos_profiles import profile_from_score
        qos_profile = profile_from_score(best_output.qos_score).value
        strategy = self._strategy_from_score(best_output.strategy_score)

        agent_id = getattr(best_agent, "agent_id", str(best_agent))

        decision = FuzzyDecision(
            agent_id=agent_id,
            agent_score=round(best_score, 2),
            qos_profile=qos_profile,
            strategy=strategy,
            all_scores=all_scores,
            inputs={"urgency": urgency, "complexity": complexity},
            inference_time_ms=round((t_end - t_start) / 1e6, 3),
        )

        logger.info(
            f"Fuzzy decision: agent={agent_id} score={best_score:.1f} "
            f"qos={qos_profile} strategy={strategy} "
            f"({decision.inference_time_ms:.1f}ms for {len(agents)} agents)"
        )

        return decision

    def _profile_bonus(self, profile: str, urgency: float, complexity: float) -> float:
        """Bonus score based on agent profile matching task characteristics."""
        if profile == "fast" and urgency >= 7 and complexity <= 4:
            return 15.0  # Fast agent bonus for urgent simple tasks
        elif profile == "quality" and complexity >= 7:
            return 15.0  # Quality agent bonus for complex tasks
        elif profile == "balanced":
            return 5.0   # Small bonus for balanced agents
        elif profile == "backup":
            return -5.0  # Penalty for backup agents (use only when needed)
        return 0.0

    @staticmethod
    def _strategy_from_score(score: float) -> str:
        """Map fuzzy strategy score (0-10) to strategy string."""
        if score <= 3.0:
            return "single"
        elif score <= 7.0:
            return "retry"
        else:
            return "fanout"

    @staticmethod
    def estimate_complexity(messages: list) -> float:
        """Estimate task complexity from message content.

        Heuristic based on total character count and message count.
        """
        if not messages:
            return 5.0

        total_chars = sum(len(m.get("content", "")) for m in messages)
        num_messages = len(messages)

        if total_chars < 50 and num_messages <= 1:
            return 2.0
        elif total_chars < 200 and num_messages <= 2:
            return 4.0
        elif total_chars < 500:
            return 6.0
        else:
            return 8.0
