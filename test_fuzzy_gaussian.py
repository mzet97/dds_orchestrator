"""
Testes das funções de pertencimento Gaussianas e estimate_complexity por features textuais.

Cobre as Fases 1, 2 e 3 do plano de implementação (proposta_modelagem_fuzzy.docx):
- Fase 1: gaussmf em todas as variáveis de entrada/saída
- Fase 2: estimate_complexity com 4 features linguísticas
- Regressão: pipeline select() completa com agentes fictícios
"""

import math
import pytest
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(agent_id, slots_idle, slots_total, latency_ms, profile="balanced"):
    a = MagicMock()
    a.agent_id = agent_id
    a.slots_idle = slots_idle
    a.slots_total = slots_total
    a.avg_latency_ms = latency_ms
    a.agent_profile = profile
    return a


def _msg(content):
    return {"role": "user", "content": content}


# ---------------------------------------------------------------------------
# Fase 1 — Gaussianas: propriedades das funções de pertencimento
# ---------------------------------------------------------------------------

class TestGaussianMembershipFunctions:

    def setup_method(self):
        from fuzzy_selector import FuzzyDecisionEngine
        self.engine = FuzzyDecisionEngine()

    def _gaussmf(self, x, c, sigma):
        return math.exp(-((x - c) ** 2) / (2 * sigma ** 2))

    def test_gaussmf_maximum_at_center(self):
        """Gaussiana atinge 1.0 no centro."""
        assert self._gaussmf(5, 5, 1.5) == pytest.approx(1.0)
        assert self._gaussmf(0, 0, 1.5) == pytest.approx(1.0)
        assert self._gaussmf(10, 10, 1.5) == pytest.approx(1.0)

    def test_gaussmf_smooth_overlap(self):
        """Termos adjacentes têm sobreposição simultânea positiva (impossível com trimf no ponto exato)."""
        # Em x=3: urgency_low (c=0, σ=1.5) e urgency_medium (c=5, σ=1.5)
        mu_low_at_3 = self._gaussmf(3, 0, 1.5)
        mu_med_at_3 = self._gaussmf(3, 5, 1.5)
        assert mu_low_at_3 > 0.0
        assert mu_med_at_3 > 0.0

    def test_gaussmf_never_zero_within_universe(self):
        """Gaussiana nunca é exatamente zero dentro do universo de discurso."""
        for x in [0, 2.5, 5, 7.5, 10]:
            mu = self._gaussmf(x, 5, 1.5)
            assert mu > 0.0

    def test_gaussmf_symmetry(self):
        """Gaussiana é simétrica em torno do centro."""
        c, sigma = 5, 1.5
        delta = 2.0
        assert self._gaussmf(c + delta, c, sigma) == pytest.approx(
            self._gaussmf(c - delta, c, sigma)
        )

    def test_engine_evaluate_returns_valid_range(self):
        """evaluate() retorna scores dentro dos universos esperados."""
        from fuzzy_selector import FuzzyInput
        out = self.engine.evaluate(FuzzyInput(
            urgency=8, complexity=6, agent_load=25, agent_latency=300
        ))
        assert 0 <= out.agent_score <= 100
        assert 0 <= out.qos_score <= 10
        assert 0 <= out.strategy_score <= 10

    def test_high_urgency_leads_to_critical_qos(self):
        """Urgência alta deve resultar em qos_score acima de 5 (próximo de critical)."""
        from fuzzy_selector import FuzzyInput
        out = self.engine.evaluate(FuzzyInput(
            urgency=9.5, complexity=5, agent_load=10, agent_latency=100
        ))
        assert out.qos_score > 5.0

    def test_low_urgency_leads_to_single_strategy(self):
        """Urgência baixa deve resultar em strategy=single."""
        from fuzzy_selector import FuzzyInput
        out = self.engine.evaluate(FuzzyInput(
            urgency=0.5, complexity=2, agent_load=10, agent_latency=100
        ))
        assert out.strategy_score <= 4.0  # região single/retry

    def test_loaded_agent_gets_lower_score(self):
        """Agente com carga alta deve receber score menor que agente livre."""
        from fuzzy_selector import FuzzyInput
        out_free = self.engine.evaluate(FuzzyInput(
            urgency=5, complexity=5, agent_load=5, agent_latency=300
        ))
        out_busy = self.engine.evaluate(FuzzyInput(
            urgency=5, complexity=5, agent_load=95, agent_latency=300
        ))
        assert out_free.agent_score > out_busy.agent_score


# ---------------------------------------------------------------------------
# Fase 2 — estimate_complexity com features textuais
# ---------------------------------------------------------------------------

class TestEstimateComplexity:

    def setup_method(self):
        from fuzzy_selector import FuzzyDecisionEngine
        self.estimate = FuzzyDecisionEngine.estimate_complexity

    def test_empty_messages_returns_default(self):
        assert self.estimate([]) == 5.0

    def test_simple_greeting_is_low(self):
        score = self.estimate([_msg("Hello")])
        assert score < 3.0, f"expected < 3.0, got {score}"

    def test_short_question_is_low_to_medium(self):
        score = self.estimate([_msg("What is DDS?")])
        assert score < 5.0, f"expected < 5.0, got {score}"

    def test_complex_prompt_is_high(self):
        text = (
            "Analise e compare as diferenças entre HTTP, gRPC e DDS para "
            "sistemas distribuídos de inferência LLM. Justifique cada escolha "
            "considerando latência, throughput e escalabilidade. Liste os passos "
            "necessários para migrar de HTTP para DDS em produção."
        )
        score = self.estimate([_msg(text)])
        # Com L_MAX=64, o prompt de ~43 tokens tem f1≈0.67; somando f4=1.0
        # (múltiplas palavras-chave) o score esperado fica em torno de 4.5–6.0.
        assert score >= 4.5, f"expected >= 4.5, got {score}"

    def test_multistep_keyword_increases_score(self):
        base = self.estimate([_msg("Explain DDS.")])
        with_keyword = self.estimate([_msg("Analyze and compare DDS versus gRPC step by step.")])
        assert with_keyword > base

    def test_long_text_scores_higher_than_short(self):
        short = self.estimate([_msg("Hi")])
        long_ = self.estimate([_msg("word " * 200)])
        assert long_ > short

    def test_score_within_valid_range(self):
        texts = [
            "Hello",
            "Analyze and compare HTTP versus DDS for distributed LLM inference.",
            "word " * 600,  # above L_MAX
        ]
        for t in texts:
            score = self.estimate([_msg(t)])
            assert 0.0 <= score <= 10.0, f"score out of range: {score} for text '{t[:40]}'"

    def test_dict_and_pydantic_messages_equivalent(self):
        """estimate_complexity aceita tanto dicts quanto objetos com .content."""
        text = "Analise e compare os protocolos."
        dict_score = self.estimate([{"role": "user", "content": text}])

        obj = MagicMock()
        obj.content = text
        obj_score = self.estimate([obj])

        assert dict_score == pytest.approx(obj_score)

    def test_portuguese_subordinate_conjunctions_increase_depth(self):
        simple = self.estimate([_msg("DDS is fast.")])
        complex_ = self.estimate([_msg(
            "DDS é rápido porque usa shared memory, embora requeira "
            "configuração cuidadosa quando há múltiplos domínios."
        )])
        assert complex_ > simple

    def test_english_subordinate_conjunctions_increase_depth(self):
        simple = self.estimate([_msg("DDS is fast.")])
        complex_ = self.estimate([_msg(
            "DDS is fast because it uses shared memory, although it requires "
            "careful configuration when multiple domains are involved."
        )])
        assert complex_ > simple


# ---------------------------------------------------------------------------
# Regressão — pipeline select() completa
# ---------------------------------------------------------------------------

class TestSelectPipeline:

    def setup_method(self):
        from fuzzy_selector import FuzzyDecisionEngine
        self.engine = FuzzyDecisionEngine()

    def test_select_single_agent_returns_decision(self):
        """select() com um agente retorna FuzzyDecision válido."""
        agent = _make_agent("a1", slots_idle=3, slots_total=4, latency_ms=200)
        task = {"urgency": 7, "messages": [_msg("Explain DDS.")]}
        decision = self.engine.select(task, [agent])

        assert decision.agent_id == "a1"
        assert decision.qos_profile in ("low_cost", "balanced", "critical")
        assert decision.strategy in ("single", "retry", "fanout")
        assert decision.inference_time_ms >= 0

    def test_select_prefers_faster_agent(self):
        """Entre agente rápido e lento com mesma carga, seleciona o mais rápido."""
        fast = _make_agent("fast", slots_idle=2, slots_total=4, latency_ms=50)
        slow = _make_agent("slow", slots_idle=2, slots_total=4, latency_ms=1800)
        task = {"urgency": 5, "messages": [_msg("Compare protocols.")]}
        decision = self.engine.select(task, [fast, slow])

        assert decision.agent_id == "fast"

    def test_select_prefers_less_loaded_agent(self):
        """Entre agente livre e saturado com mesma latência, seleciona o livre."""
        free = _make_agent("free", slots_idle=4, slots_total=4, latency_ms=300)
        busy = _make_agent("busy", slots_idle=0, slots_total=4, latency_ms=300)
        task = {"urgency": 5, "messages": [_msg("Process request.")]}
        decision = self.engine.select(task, [free, busy])

        assert decision.agent_id == "free"

    def test_select_empty_agents_returns_empty(self):
        """select() com lista vazia retorna decision com agent_id vazio."""
        task = {"urgency": 5, "messages": [_msg("Hello.")]}
        decision = self.engine.select(task, [])
        assert decision.agent_id == ""

    def test_select_explicit_complexity_bypasses_estimate(self):
        """Se complexity já é fornecido no task_input, não chama estimate_complexity."""
        agent = _make_agent("a1", slots_idle=2, slots_total=4, latency_ms=300)
        task = {"urgency": 5, "complexity": 9.0, "messages": [_msg("Hi.")]}
        decision = self.engine.select(task, [agent])
        # Complexity 9.0 é alta → qos_profile deve ser balanced ou critical
        assert decision.qos_profile in ("balanced", "critical")

    def test_all_scores_populated(self):
        """all_scores deve conter entrada para cada agente avaliado."""
        agents = [
            _make_agent("a1", slots_idle=2, slots_total=4, latency_ms=200),
            _make_agent("a2", slots_idle=1, slots_total=4, latency_ms=500),
            _make_agent("a3", slots_idle=0, slots_total=4, latency_ms=900),
        ]
        task = {"urgency": 6, "messages": [_msg("Analyze this.")]}
        decision = self.engine.select(task, agents)
        assert set(decision.all_scores.keys()) == {"a1", "a2", "a3"}
