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

# Top-level import (was previously inside select() — move out of hot path
# and avoid silent ImportError on first request).
try:
    from qos_profiles import profile_from_score
except ImportError:
    profile_from_score = None
    logger.warning("qos_profiles module not available; QoS mapping will use 'balanced' fallback")


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
        """Build the fuzzy control system with 4 inputs, 3 outputs, 18 rules.

        Fundamentação teórica:
        - Conjuntos fuzzy e Princípio de Extensão de Zadeh (1975):
          μ_B̃(y) = sup{ min(μ_Ã₁(x₁),...,μ_Ãₙ(xₙ)) : f(x₁,...,xₙ)=y }
          O operador AND (mínimo) nas regras Mamdani realiza esta t-norma.
        - Funções de pertencimento Gaussianas (Mendel, 2001):
          gaussmf(x; c, σ) = exp(−(x−c)² / (2σ²))
          Substituem as triangulares para garantir superfícies de controle
          infinitamente diferenciáveis e sem descontinuidades de gradiente.
        - Defuzzificação por centroide (método padrão do skfuzzy):
          y* = Σ y_j·μ_out(y_j) / Σ μ_out(y_j)
        """

        # ─── Input variables (Antecedents) ───────────────────────────────
        urgency = ctrl.Antecedent(np.arange(0, 11, 0.5), 'urgency')
        complexity = ctrl.Antecedent(np.arange(0, 11, 0.5), 'complexity')
        agent_load = ctrl.Antecedent(np.arange(0, 101, 1), 'agent_load')
        agent_latency = ctrl.Antecedent(np.arange(0, 2001, 10), 'agent_latency')

        # ─── Output variables (Consequents) ──────────────────────────────
        agent_score = ctrl.Consequent(np.arange(0, 101, 1), 'agent_score')
        qos_profile = ctrl.Consequent(np.arange(0, 11, 0.5), 'qos_profile')
        strategy = ctrl.Consequent(np.arange(0, 11, 0.5), 'strategy')

        # ─── Membership functions (Gaussianas) ───────────────────────────
        #
        # Fundamentação: Princípio de Extensão de Zadeh (1975).
        # A imagem fuzzy de Ã sob f é B̃ com μ_B̃(y) = sup{μ_Ã(x) : f(x)=y}.
        # O operador AND (mínimo) nas regras realiza a t-norma sobre os graus
        # de pertencimento, conforme derivado do princípio de extensão multivariado.
        #
        # Forma: gaussmf(x; c, σ) = exp(−(x−c)² / (2σ²))
        # Vantagem sobre trimf: infinitamente diferenciável, sem descontinuidades
        # de gradiente nos vértices → superfícies de controle suaves (Mendel, 2001).

        # Urgency [0, 10]: c ∈ {0, 5, 10}, σ = 1.5
        urgency['low']    = fuzz.gaussmf(urgency.universe, 0,  1.5)
        urgency['medium'] = fuzz.gaussmf(urgency.universe, 5,  1.5)
        urgency['high']   = fuzz.gaussmf(urgency.universe, 10, 1.5)

        # Complexity [0, 10]: mesmos parâmetros de urgency (universo simétrico)
        complexity['low']    = fuzz.gaussmf(complexity.universe, 0,  1.5)
        complexity['medium'] = fuzz.gaussmf(complexity.universe, 5,  1.5)
        complexity['high']   = fuzz.gaussmf(complexity.universe, 10, 1.5)

        # Agent load [0, 100]: c ∈ {0, 50, 100}, σ = 15
        agent_load['low']    = fuzz.gaussmf(agent_load.universe, 0,   15)
        agent_load['medium'] = fuzz.gaussmf(agent_load.universe, 50,  15)
        agent_load['high']   = fuzz.gaussmf(agent_load.universe, 100, 15)

        # Agent latency [0, 2000ms]: c ∈ {0, 600, 2000}, σ ∈ {100, 200, 250}
        agent_latency['low']    = fuzz.gaussmf(agent_latency.universe, 0,    100)
        agent_latency['medium'] = fuzz.gaussmf(agent_latency.universe, 600,  200)
        agent_latency['high']   = fuzz.gaussmf(agent_latency.universe, 2000, 250)

        # Agent score [0, 100]: c ∈ {0, 50, 100}, σ = 15
        agent_score['low']    = fuzz.gaussmf(agent_score.universe, 0,   15)
        agent_score['medium'] = fuzz.gaussmf(agent_score.universe, 50,  15)
        agent_score['high']   = fuzz.gaussmf(agent_score.universe, 100, 15)

        # QoS profile [0, 10]: c ∈ {0, 5, 10}, σ = 1.5
        qos_profile['low_cost'] = fuzz.gaussmf(qos_profile.universe, 0,  1.5)
        qos_profile['balanced'] = fuzz.gaussmf(qos_profile.universe, 5,  1.5)
        qos_profile['critical'] = fuzz.gaussmf(qos_profile.universe, 10, 1.5)

        # Strategy [0, 10]: c ∈ {0, 5, 10}, σ = 1.5
        strategy['single'] = fuzz.gaussmf(strategy.universe, 0,  1.5)
        strategy['retry']  = fuzz.gaussmf(strategy.universe, 5,  1.5)
        strategy['fanout'] = fuzz.gaussmf(strategy.universe, 10, 1.5)

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

        Pipeline Mamdani (4 etapas):
          1. Fuzzificação   — gaussmf para cada variável de entrada
          2. Avaliação      — força de disparo w_i = min(μ_A(x₁), μ_B(x₂), ...)
          3. Agregação      — μ_out(y) = max_i[ min(w_i, μ_Ci(y)) ]
          4. Defuzzificação — centroide: y* = Σ y·μ_out / Σ μ_out

        Cria uma simulação nova a partir do ControlSystem compartilhado (thread-safe).
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
        if profile_from_score is not None:
            qos_profile = profile_from_score(best_output.qos_score).value
        else:
            qos_profile = "balanced"
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
        """Estimate task complexity from message text using four linguistic features.

        Features (Zadeh Extension applied to prompt space → complexity score):
          f1 — token count normalized by L_max=512          (weight 0.4)
          f2 — entity density (capitalized words mid-sentence) (weight 0.2)
          f3 — syntactic depth via subordinate conjunctions   (weight 0.2)
          f4 — multi-step reasoning keyword indicators        (weight 0.2)

        Returns a value in [0, 10].
        Accepts both dict messages and Pydantic ChatMessage objects.
        """
        if not messages:
            return 5.0

        def _content(m):
            if isinstance(m, dict):
                return m.get("content", "")
            return getattr(m, "content", "") or ""

        # Concatenate all user/assistant message content
        full_text = " ".join(_content(m) for m in messages)
        if not full_text.strip():
            return 5.0

        tokens = full_text.split()
        n_tok = len(tokens)
        # L_MAX calibrado para prompts típicos neste sistema (8–100 tokens).
        # Prompts do PROMPT_BANK variam de 16 a ~100 tokens; L_MAX=64 cobre
        # o comprimento médio de referência sem penalizar textos curtos mas ricos.
        L_MAX = 64

        # f1: token count normalized
        f1 = min(n_tok / L_MAX, 1.0)

        # f2: entity density — capitalized words that are NOT at sentence start
        sentences = full_text.replace("?", ".").replace("!", ".").split(".")
        entity_count = 0
        for sent in sentences:
            words = sent.strip().split()
            # skip first word of sentence (always capitalized)
            for w in words[1:]:
                if w and w[0].isupper() and w.isalpha():
                    entity_count += 1
        f2 = entity_count / (n_tok + 1)

        # f3: syntactic depth via subordinate conjunctions
        _subordinates_pt = {"que", "porque", "embora", "quando", "se", "enquanto",
                             "embora", "pois", "como", "conforme", "apesar"}
        _subordinates_en = {"that", "because", "although", "when", "if", "while",
                             "since", "unless", "whether", "though", "as"}
        subordinates = _subordinates_pt | _subordinates_en
        sub_count = sum(1 for w in tokens if w.lower().rstrip(".,;:") in subordinates)
        f3 = min(sub_count / (n_tok / 10 + 1), 1.0)

        # f4: multi-step reasoning indicators (weighted keywords)
        _keywords = {
            # Portuguese
            "analise": 1.0, "analisar": 1.0, "compare": 1.0, "comparar": 1.0,
            "explique": 0.5, "explicar": 0.5, "justifique": 0.5, "justificar": 0.5,
            "descreva": 0.5, "liste": 0.5, "enumere": 0.5, "detalhe": 0.5,
            "calcule": 1.0, "resolva": 1.0, "demonstre": 1.0, "prove": 1.0,
            # English
            "analyze": 1.0, "analyse": 1.0, "explain": 0.5, "compare": 1.0,
            "justify": 0.5, "describe": 0.5, "list": 0.5, "enumerate": 0.5,
            "calculate": 1.0, "solve": 1.0, "demonstrate": 1.0, "prove": 1.0,
            "step": 0.5, "steps": 0.5,
        }
        # Also detect multi-word phrases
        text_lower = full_text.lower()
        phrase_score = 0.0
        for phrase, w in [("liste os passos", 1.0), ("step by step", 1.0),
                          ("passo a passo", 1.0), ("etapa por etapa", 1.0)]:
            if phrase in text_lower:
                phrase_score += w

        word_score = sum(_keywords.get(w.lower().rstrip(".,;:?!"), 0.0) for w in tokens)
        f4 = min(word_score + phrase_score, 1.0)

        complexity = 10.0 * (0.4 * f1 + 0.2 * f2 + 0.2 * f3 + 0.2 * f4)
        return round(min(complexity, 10.0), 2)
