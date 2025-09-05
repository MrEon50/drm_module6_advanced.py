"""drm_module6_advanced.py
Advanced Dynamic Rule Matrix (DRM) v6 - Comprehensive Rule Architecture

GŁÓWNE ULEPSZENIA W WERSJI 6:
=============================

1. ZAAWANSOWANA ARCHITEKTURA REGUŁ:
   - Elastyczne typy reguł: LOGICAL (niezmienne), HEURISTIC (adaptacyjne), HYBRID (uczące się)
   - System tagowania i kategoryzacji dla lepszej organizacji
   - Metadane i proweniencja dla pełnej traceability

2. INTELIGENTNE ROZRÓŻNIANIE REGUŁ:
   - Semantyczne podobieństwo (typy, kategorie, tagi, parametry, warunki)
   - Behawioralne podobieństwo (wydajność, wzorce użycia, wskaźniki sukcesu)
   - Strukturalne podobieństwo (architektura, zależności)

3. ZAAWANSOWANA WALIDACJA:
   - Wielopoziomowa walidacja: syntaktyczna, semantyczna, statystyczna
   - Różne strategie dla różnych typów reguł
   - Automatyczna kwarantanna i odzyskiwanie

4. SYSTEM SELEKCJI I FILTROWANIA:
   - Wyszukiwanie po tagach (AND/OR logic)
   - Filtrowanie po kategorii, typie, wydajności, użyciu
   - Zaawansowane zapytania z wieloma kryteriami
   - Indeksowanie dla szybkiego wyszukiwania

5. DETEKCJA DUPLIKATÓW I PODOBIEŃSTWA:
   - Automatyczna detekcja duplikatów semantycznych i behawioralnych
   - Rekomendacje dotyczące łączenia/usuwania reguł
   - Cache podobieństwa dla wydajności

6. ANALIZA WZORCÓW I TRENDÓW:
   - Analiza trendów wydajności w czasie
   - Wykrywanie wzorców użycia
   - Statystyki rozkładu i korelacji
   - Rekomendacje optymalizacji

7. ELASTYCZNOŚĆ BEZ UTRATY LOGICZNOŚCI:
   - Ograniczenia parametrów z walidacją
   - Grupy konfliktów dla spójności
   - Priorytety i strategie rozwiązywania konfliktów
   - Automatyczne sprawdzanie spójności

PRZYKŁAD UŻYCIA:
===============

```python
# Tworzenie systemu DRM v6
drm = DRMSystem()

# Dodawanie reguły z tagami i kategorią
rule = Rule(
    id="physics_conservation",
    name="Mass Conservation Law",
    rtype=LOGICAL,
    category=RULE_CATEGORIES["PHYSICS"],
    tags={"conservation", "physics", "fundamental"},
    priority=10,
    pre_conditions=["mass_initial", "domain_defined"],
    post_conditions=["mass_conserved"],
    params={"tolerance": RuleParameter(1e-6, min_val=1e-9, max_val=1e-3)},
    tests=[lambda rule, ctx: abs(ctx.get("mass_change", 1.0)) < rule.params["tolerance"].value]
)

drm.add_rule(rule)

# Zaawansowane wyszukiwanie
physics_rules = drm.search_rules_by_tags(["physics", "conservation"], match_all=True)
high_perf_rules = drm.search_rules_by_performance(min_performance=0.8)

# Filtrowanie z wieloma kryteriami
filtered = drm.filter_rules(
    rule_types=[LOGICAL, HYBRID],
    categories=["physics", "mathematics"],
    min_performance=0.7,
    performance_classes=["EXCELLENT", "GOOD"]
)

# Analiza duplikatów
duplicates = drm.detect_duplicates(semantic_threshold=0.8)
print(f"Znaleziono {duplicates['total_duplicates']} potencjalnych duplikatów")

# Analiza wzorców
patterns = drm.analyze_rule_patterns()
print(f"Średnia wydajność: {patterns['performance_stats']['mean']:.3f}")

# Rekomendacje dla reguły
recommendations = drm.get_rule_recommendations("physics_conservation")
print(f"Status reguły: {recommendations['overall_health']}")
```

ARCHITEKTURA ELASTYCZNOŚCI:
==========================

1. PARAMETRY Z OGRANICZENIAMI:
   - min_val, max_val dla zakresu wartości
   - constraint_fn dla niestandardowych ograniczeń
   - Automatyczna walidacja i clamp

2. GRUPY KONFLIKTÓW:
   - Logiczne grupowanie wzajemnie wykluczających się reguł
   - Automatyczne rozwiązywanie konfliktów
   - Strategie: priorytet, wydajność, konsensus

3. TYPY REGUŁ Z RÓŻNĄ ELASTYCZNOŚCIĄ:
   - LOGICAL: Niezmienne, ścisła walidacja
   - HEURISTIC: Adaptacyjne, tolerancyjne
   - HYBRID: Uczące się, balans między sztywnością a elastycznością

4. SYSTEM TAGOWANIA:
   - Elastyczne kategoryzowanie
   - Wyszukiwanie semantyczne
   - Automatyczne sugerowanie tagów

Ten moduł zapewnia spójną architekturę reguł z zachowaniem elastyczności
poprzez inteligentne ograniczenia, walidację i system konfliktów.
"""

# Import podstawowych modułów
from __future__ import annotations
import math, random, json, time, warnings
from typing import Dict, Any, List, Optional, Callable, Tuple, Set, Union
from collections import deque, defaultdict
from dataclasses import dataclass
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import numpy as np
    HAS_NUMPY = True
except Exception:
    np = None
    HAS_NUMPY = False

EPS = 1e-9

# Rule types
LOGICAL = "LOGICAL"
HEURISTIC = "HEURISTIC"
HYBRID = "HYBRID"

# Rule categories for better organization
RULE_CATEGORIES = {
    "PHYSICS": "physics",
    "MATH": "mathematics",
    "LOGIC": "logical_reasoning",
    "OPTIMIZATION": "optimization",
    "PATTERN": "pattern_recognition",
    "CONTROL": "control_systems",
    "LEARNING": "machine_learning",
    "DOMAIN": "domain_specific",
    "UTILITY": "utility_functions",
    "EXPERIMENTAL": "experimental"
}

# Performance thresholds for rule classification
PERFORMANCE_THRESHOLDS = {
    "EXCELLENT": 0.9,
    "GOOD": 0.7,
    "AVERAGE": 0.5,
    "POOR": 0.3,
    "FAILING": 0.1
}

# HIERARCHIA ZŁOŻONOŚCI - KLUCZOWA INNOWACJA DRM v6!
COMPLEXITY_LEVELS = {
    "ATOMIC": 1,      # Pojedyncze operacje, proste warunki
    "SIMPLE": 2,      # Kombinacje atomowych, podstawowa logika
    "COMPOUND": 3,    # Złożone warunki, wieloetapowe
    "COMPLEX": 4,     # Zaawansowane algorytmy, uczenie
    "COMPOSITE": 5    # Hierarchie reguł, meta-reguły
}

# Thresholds for automatic complexity classification
COMPLEXITY_THRESHOLDS = {
    "max_conditions": [1, 3, 7, 15, float('inf')],
    "max_parameters": [1, 3, 8, 20, float('inf')],
    "max_tests": [1, 2, 5, 10, float('inf')],
    "max_dependencies": [0, 1, 3, 8, float('inf')]
}

# Parameter types for better type safety
@dataclass
class RuleParameter:
    """Type-safe rule parameter with constraints and differentiation info"""
    value: float
    param_type: str = "float"
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    requires_grad: bool = False
    constraint_fn: Optional[Callable[[float], bool]] = None

    def __post_init__(self):
        self.validate()

    def validate(self) -> bool:
        """Validate parameter constraints"""
        if self.min_val is not None and self.value < self.min_val - EPS:
            return False
        if self.max_val is not None and self.value > self.max_val + EPS:
            return False
        if self.constraint_fn and not self.constraint_fn(self.value):
            return False
        return True

    def clamp(self):
        """Clamp value to constraints"""
        if self.min_val is not None:
            self.value = max(self.min_val, self.value)
        if self.max_val is not None:
            self.value = min(self.max_val, self.value)

# ------------------ ComplexityAnalyzer ------------------
class ComplexityAnalyzer:
    """Analizator złożoności reguł"""

    @staticmethod
    def calculate_complexity_score(rule: 'Rule') -> float:
        """Oblicza numeryczny wskaźnik złożoności reguły"""
        score = 0.0

        # Złożoność warunków (30% wagi)
        condition_count = len(rule.pre_conditions) + len(rule.post_conditions)
        score += (condition_count * 0.3)

        # Złożoność parametrów (25% wagi)
        param_complexity = 0
        for param in rule.params.values():
            param_complexity += 1
            if param.constraint_fn is not None:
                param_complexity += 0.5  # Niestandardowe ograniczenia
            if param.requires_grad:
                param_complexity += 0.3  # Parametry uczące się
        score += (param_complexity * 0.25)

        # Złożoność testów (20% wagi)
        test_count = len(rule.tests)
        score += (test_count * 0.2)

        # Złożoność typu (15% wagi)
        type_complexity = {
            LOGICAL: 0.5,    # Proste ale sztywne
            HEURISTIC: 1.0,  # Średnie
            HYBRID: 1.5      # Złożone (uczenie)
        }
        score += (type_complexity.get(rule.type, 1.0) * 0.15)

        # Złożoność metadanych (10% wagi)
        metadata_complexity = len(rule.metadata) * 0.1
        score += (metadata_complexity * 0.1)

        return score

    @staticmethod
    def classify_complexity(rule: 'Rule') -> str:
        """Klasyfikuje regułę do poziomu złożoności"""
        score = ComplexityAnalyzer.calculate_complexity_score(rule)

        # Mapowanie wyniku na poziomy
        if score <= 1.0:
            return "ATOMIC"
        elif score <= 2.5:
            return "SIMPLE"
        elif score <= 5.0:
            return "COMPOUND"
        elif score <= 8.0:
            return "COMPLEX"
        else:
            return "COMPOSITE"

    @staticmethod
    def get_execution_priority(complexity_level: str) -> int:
        """Zwraca priorytet wykonania (niższe = szybsze)"""
        priorities = {
            "ATOMIC": 1,     # Najszybsze
            "SIMPLE": 2,
            "COMPOUND": 3,
            "COMPLEX": 4,
            "COMPOSITE": 5   # Najwolniejsze
        }
        return priorities.get(complexity_level, 3)

# ------------------ Enhanced Rule with Hierarchy ------------------
class Rule:
    def __init__(self,
                 id: str,
                 name: Optional[str] = None,
                 rtype: str = HEURISTIC,
                 init_weight: float = 1.0,
                 init_mean: float = 0.5,
                 init_var: float = 0.25,
                 latent_z: Optional[List[float]] = None,
                 pre_conditions: Optional[List[str]] = None,
                 post_conditions: Optional[List[str]] = None,
                 params: Optional[Dict[str, Union[Dict[str, Any], RuleParameter]]] = None,
                 tests: Optional[List[Callable[['Rule', Dict[str, Any]], bool]]] = None,
                 provenance: Optional[Dict[str, Any]] = None,
                 priority: int = 0,
                 conflict_groups: Optional[Set[str]] = None,
                 tags: Optional[Set[str]] = None,
                 category: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 # Hierarchia złożoności - nowe parametry v6
                 complexity_level: Optional[str] = None,
                 parent_rules: Optional[List[str]] = None,
                 child_rules: Optional[List[str]] = None,
                 execution_cost: Optional[float] = None):

        self.id = id
        self.name = name or id
        self.type = rtype
        self.priority = priority  # Higher priority rules win conflicts
        self.conflict_groups = conflict_groups or set()

        # Enhanced v6 features
        self.tags = tags or set()
        self.category = category or RULE_CATEGORIES["UTILITY"]
        self.metadata = metadata or {}
        self.similarity_cache = {}  # Cache for similarity calculations

        # Hierarchia złożoności - v6
        self.complexity_level = complexity_level
        self.parent_rules = parent_rules or []  # Reguły nadrzędne
        self.child_rules = child_rules or []    # Reguły podrzędne
        self.execution_cost = execution_cost or 1.0

        # probabilistic
        self.weight = float(max(EPS, init_weight))
        self.post_mean = float(init_mean)
        self.post_var = float(init_var)
        self.observations = 0
        self.usage_count = 0
        self.is_new = True
        self.quarantined = False
        self.quarantine_reason = None
        self.latent_z = list(latent_z) if latent_z is not None else None

        # semantic
        self.pre_conditions = list(pre_conditions) if pre_conditions else []
        self.post_conditions = list(post_conditions) if post_conditions else []

        # Enhanced parameters handling
        self.params: Dict[str, RuleParameter] = {}
        if params:
            for k, v in params.items():
                if isinstance(v, RuleParameter):
                    self.params[k] = v
                elif isinstance(v, dict):
                    # Convert legacy dict format
                    self.params[k] = RuleParameter(
                        value=v.get("value", 0.0),
                        param_type=v.get("type", "float"),
                        min_val=v.get("min"),
                        max_val=v.get("max"),
                        requires_grad=v.get("requires_grad", self.type == HYBRID)
                    )

        # Tests
        self.tests = list(tests) if tests else []
        self.provenance = dict(provenance) if provenance else {}
        self.created_at = time.time()

        # Enhanced diagnostics
        self.history = deque(maxlen=200)
        self.activation_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.last_activated = None

        # Automatyczna klasyfikacja złożoności jeśli nie podano
        if self.complexity_level is None:
            self.complexity_level = ComplexityAnalyzer.classify_complexity(self)

        # Oblicz koszt wykonania
        if execution_cost is None:
            self.execution_cost = self._calculate_execution_cost()

        # Validate on creation
        self._validate_construction()

    def _validate_construction(self):
        """Validate rule construction"""
        if self.type == LOGICAL and not self.tests:
            warnings.warn(f"LOGICAL rule {self.id} has no tests - may not be properly validated")

        # Validate all parameters
        for name, param in self.params.items():
            if not param.validate():
                logger.warning(f"Parameter {name} in rule {self.id} violates constraints")

    def _calculate_execution_cost(self) -> float:
        """Oblicza szacowany koszt wykonania reguły"""
        base_cost = COMPLEXITY_LEVELS[self.complexity_level]

        # Dodatkowe koszty
        condition_cost = (len(self.pre_conditions) + len(self.post_conditions)) * 0.1
        param_cost = len(self.params) * 0.05
        test_cost = len(self.tests) * 0.2

        return base_cost + condition_cost + param_cost + test_cost

    def can_compose_with(self, other: 'Rule') -> bool:
        """Sprawdza czy można komponować z inną regułą"""
        # Nie można komponować z regułami tego samego poziomu COMPOSITE
        if (self.complexity_level == "COMPOSITE" and
            other.complexity_level == "COMPOSITE"):
            return False

        # Sprawdź konflikty
        if self.conflict_groups & other.conflict_groups:
            return False

        # Sprawdź kompatybilność warunków
        if set(self.post_conditions) & set(other.pre_conditions):
            return True  # Można łączyć w pipeline

        return True

    def get_hierarchy_info(self) -> Dict[str, Any]:
        """Zwraca informacje o hierarchii"""
        return {
            "complexity_level": self.complexity_level,
            "complexity_score": ComplexityAnalyzer.calculate_complexity_score(self),
            "execution_cost": self.execution_cost,
            "execution_priority": ComplexityAnalyzer.get_execution_priority(self.complexity_level),
            "parent_count": len(self.parent_rules),
            "child_count": len(self.child_rules),
            "can_be_composed": self.complexity_level != "COMPOSITE"
        }

    # Enhanced v6 methods for tags and categorization
    def add_tag(self, tag: str) -> bool:
        """Add a tag to the rule"""
        if isinstance(tag, str) and tag.strip():
            self.tags.add(tag.strip().lower())
            return True
        return False

    def remove_tag(self, tag: str) -> bool:
        """Remove a tag from the rule"""
        tag_lower = tag.strip().lower()
        if tag_lower in self.tags:
            self.tags.remove(tag_lower)
            return True
        return False

    def has_tag(self, tag: str) -> bool:
        """Check if rule has a specific tag"""
        return tag.strip().lower() in self.tags

    def has_any_tags(self, tags: List[str]) -> bool:
        """Check if rule has any of the specified tags"""
        return any(self.has_tag(tag) for tag in tags)

    def has_all_tags(self, tags: List[str]) -> bool:
        """Check if rule has all of the specified tags"""
        return all(self.has_tag(tag) for tag in tags)

    def set_category(self, category: str) -> bool:
        """Set rule category"""
        if category in RULE_CATEGORIES.values():
            self.category = category
            return True
        return False

    def get_performance_class(self) -> str:
        """Get performance classification"""
        if self.post_mean >= PERFORMANCE_THRESHOLDS["EXCELLENT"]:
            return "EXCELLENT"
        elif self.post_mean >= PERFORMANCE_THRESHOLDS["GOOD"]:
            return "GOOD"
        elif self.post_mean >= PERFORMANCE_THRESHOLDS["AVERAGE"]:
            return "AVERAGE"
        elif self.post_mean >= PERFORMANCE_THRESHOLDS["POOR"]:
            return "POOR"
        else:
            return "FAILING"

    # Bayesian posterior update
    def update_posterior(self, reward: Optional[float], obs_var: float = 0.05, context: Optional[Dict] = None):
        if reward is None:
            return

        self.observations += 1
        self.usage_count += 1
        self.is_new = False

        # Track success/failure
        if reward > 0.5:  # Threshold for success
            self.success_count += 1
        else:
            self.failure_count += 1

        prior_prec = 1.0 / max(EPS, self.post_var)
        like_prec = 1.0 / max(EPS, obs_var)
        post_var = 1.0 / (prior_prec + like_prec)
        post_mean = post_var * (self.post_mean * prior_prec + reward * like_prec)

        self.post_mean = float(post_mean)
        self.post_var = float(post_var)

        # Enhanced history with context info
        self.history.append(("update", reward, self.post_mean, self.post_var, context))

    def get_success_rate(self) -> float:
        """Get empirical success rate"""
        total = self.success_count + self.failure_count
        return self.success_count / max(1, total)

    def to_dict(self) -> Dict[str, Any]:
        """Enhanced serialization with v6 features"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "priority": self.priority,
            "conflict_groups": list(self.conflict_groups),
            "tags": list(self.tags),  # v6 addition
            "category": self.category,  # v6 addition
            "metadata": dict(self.metadata),  # v6 addition
            "complexity_level": self.complexity_level,  # v6 addition
            "parent_rules": list(self.parent_rules),  # v6 addition
            "child_rules": list(self.child_rules),  # v6 addition
            "execution_cost": self.execution_cost,  # v6 addition
            "weight": self.weight,
            "post_mean": self.post_mean,
            "post_var": self.post_var,
            "observations": self.observations,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "quarantined": self.quarantined,
            "quarantine_reason": self.quarantine_reason,
            "latent_z": list(self.latent_z) if self.latent_z is not None else None,
            "pre_conditions": list(self.pre_conditions),
            "post_conditions": list(self.post_conditions),
            "params": {k: {
                "value": v.value,
                "type": v.param_type,
                "min": v.min_val,
                "max": v.max_val,
                "requires_grad": v.requires_grad
            } for k, v in self.params.items()},
            "provenance": dict(self.provenance),
            "created_at": self.created_at
        }

# ------------------ Advanced Rule Analytics v6 ------------------
class RuleAnalyzer:
    """Advanced rule analysis and pattern recognition system"""

    def __init__(self):
        self.similarity_cache = {}
        self.pattern_cache = {}

    def compute_semantic_similarity(self, rule_a: Rule, rule_b: Rule) -> float:
        """Compute semantic similarity between two rules"""
        cache_key = tuple(sorted([rule_a.id, rule_b.id]))
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]

        similarity = 0.0

        # Type similarity (30% weight)
        if rule_a.type == rule_b.type:
            similarity += 0.3

        # Category similarity (20% weight)
        if rule_a.category == rule_b.category:
            similarity += 0.2

        # Tag similarity (25% weight)
        if rule_a.tags and rule_b.tags:
            tag_intersection = len(rule_a.tags & rule_b.tags)
            tag_union = len(rule_a.tags | rule_b.tags)
            if tag_union > 0:
                similarity += 0.25 * (tag_intersection / tag_union)

        # Parameter similarity (15% weight)
        common_params = set(rule_a.params.keys()) & set(rule_b.params.keys())
        if common_params:
            param_sim = 0.0
            for param in common_params:
                val_a = rule_a.params[param].value
                val_b = rule_b.params[param].value
                # Normalized difference
                max_val = max(abs(val_a), abs(val_b), 1.0)
                param_sim += 1.0 - (abs(val_a - val_b) / max_val)
            similarity += 0.15 * (param_sim / len(common_params))

        # Condition similarity (10% weight)
        pre_sim = self._condition_similarity(rule_a.pre_conditions, rule_b.pre_conditions)
        post_sim = self._condition_similarity(rule_a.post_conditions, rule_b.post_conditions)
        similarity += 0.05 * pre_sim + 0.05 * post_sim

        self.similarity_cache[cache_key] = similarity
        return similarity

    def _condition_similarity(self, cond_a: List[str], cond_b: List[str]) -> float:
        """Compute similarity between condition lists"""
        if not cond_a and not cond_b:
            return 1.0
        if not cond_a or not cond_b:
            return 0.0

        set_a = set(cond_a)
        set_b = set(cond_b)
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)

        return intersection / union if union > 0 else 0.0

    def compute_behavioral_similarity(self, rule_a: Rule, rule_b: Rule) -> float:
        """Compute behavioral similarity based on performance patterns"""
        # Performance similarity
        perf_diff = abs(rule_a.post_mean - rule_b.post_mean)
        perf_sim = 1.0 - min(perf_diff, 1.0)

        # Usage pattern similarity
        usage_a = rule_a.usage_count / max(1, rule_a.observations)
        usage_b = rule_b.usage_count / max(1, rule_b.observations)
        usage_diff = abs(usage_a - usage_b)
        usage_sim = 1.0 - min(usage_diff, 1.0)

        # Success rate similarity
        success_a = rule_a.get_success_rate()
        success_b = rule_b.get_success_rate()
        success_diff = abs(success_a - success_b)
        success_sim = 1.0 - min(success_diff, 1.0)

        return (perf_sim * 0.5 + usage_sim * 0.25 + success_sim * 0.25)

    def find_duplicates(self, rules: Dict[str, Rule],
                       semantic_threshold: float = 0.8,
                       behavioral_threshold: float = 0.9) -> List[Tuple[str, str, float, str]]:
        """Find potential duplicate rules"""
        duplicates = []
        rule_list = list(rules.values())

        for i in range(len(rule_list)):
            for j in range(i + 1, len(rule_list)):
                rule_a, rule_b = rule_list[i], rule_list[j]

                semantic_sim = self.compute_semantic_similarity(rule_a, rule_b)
                behavioral_sim = self.compute_behavioral_similarity(rule_a, rule_b)

                if semantic_sim >= semantic_threshold:
                    duplicates.append((rule_a.id, rule_b.id, semantic_sim, "semantic"))
                elif behavioral_sim >= behavioral_threshold:
                    duplicates.append((rule_a.id, rule_b.id, behavioral_sim, "behavioral"))

        return duplicates

# ------------------ Enhanced DRMSystem with Hierarchy ------------------
class DRMSystem:
    def __init__(self):
        self.rules: Dict[str, Rule] = {}
        self.archived: Dict[str, Rule] = {}
        self.cycle_count = 0

        # Enhanced v6 components
        self.analyzer = RuleAnalyzer()
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)  # tag -> rule_ids
        self.category_index: Dict[str, Set[str]] = defaultdict(set)  # category -> rule_ids
        self.complexity_index: Dict[str, Set[str]] = defaultdict(set)  # complexity -> rule_ids
        self.execution_queue: List[Tuple[str, float]] = []  # (rule_id, cost)

        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.diversity_history = deque(maxlen=1000)
        self.audit_log: List[Dict[str, Any]] = []

    def audit(self, action: str, rule_id: str, details: Dict[str, Any]):
        """Add audit entry"""
        self.audit_log.append({
            "timestamp": time.time(),
            "action": action,
            "rule_id": rule_id,
            "details": details
        })

    def add_rule(self, rule: Rule, check_conflicts: bool = True) -> Dict[str, Any]:
        """Enhanced rule addition with conflict checking"""
        result = {"success": True, "conflicts": [], "warnings": []}

        if rule.id in self.rules:
            result["success"] = False
            result["warnings"].append(f"Rule {rule.id} already exists")
            return result

        self.rules[rule.id] = rule
        rule.weight = max(EPS, rule.weight)

        # Update indexes
        self._update_indexes_for_rule(rule)

        self.audit("add", rule.id, {
            "type": rule.type,
            "priority": rule.priority,
            "complexity": rule.complexity_level
        })

        return result

    def _update_indexes_for_rule(self, rule: Rule):
        """Update tag, category and complexity indexes for a rule"""
        # Update tag index
        for tag in rule.tags:
            self.tag_index[tag].add(rule.id)

        # Update category index
        self.category_index[rule.category].add(rule.id)

        # Update complexity index
        self.complexity_index[rule.complexity_level].add(rule.id)

    def get_rules_by_complexity(self, complexity_level: str) -> Dict[str, Rule]:
        """Zwraca reguły o określonym poziomie złożoności"""
        rule_ids = self.complexity_index.get(complexity_level, set())
        return {rid: self.rules[rid] for rid in rule_ids if rid in self.rules}

    def search_rules_by_tags(self, tags: List[str], match_all: bool = False) -> Dict[str, Rule]:
        """Search rules by tags"""
        if not tags:
            return {}

        if match_all:
            # Find rules that have ALL specified tags
            rule_ids = None
            for tag in tags:
                tag_rules = self.tag_index.get(tag.lower(), set())
                if rule_ids is None:
                    rule_ids = tag_rules.copy()
                else:
                    rule_ids &= tag_rules

            return {rid: self.rules[rid] for rid in (rule_ids or set()) if rid in self.rules}
        else:
            # Find rules that have ANY of the specified tags
            rule_ids = set()
            for tag in tags:
                rule_ids.update(self.tag_index.get(tag.lower(), set()))

            return {rid: self.rules[rid] for rid in rule_ids if rid in self.rules}

    def search_rules_by_type(self, rule_type: str) -> Dict[str, Rule]:
        """Search rules by type"""
        return {k: v for k, v in self.rules.items() if v.type == rule_type}

    def filter_rules(self,
                    rule_types: Optional[List[str]] = None,
                    categories: Optional[List[str]] = None,
                    tags: Optional[List[str]] = None,
                    match_all_tags: bool = False,
                    complexity_levels: Optional[List[str]] = None) -> Dict[str, Rule]:
        """Advanced rule filtering with multiple criteria"""

        filtered_rules = dict(self.rules)

        # Filter by types
        if rule_types:
            filtered_rules = {k: v for k, v in filtered_rules.items() if v.type in rule_types}

        # Filter by categories
        if categories:
            filtered_rules = {k: v for k, v in filtered_rules.items() if v.category in categories}

        # Filter by complexity levels
        if complexity_levels:
            filtered_rules = {k: v for k, v in filtered_rules.items() if v.complexity_level in complexity_levels}

        # Filter by tags
        if tags:
            if match_all_tags:
                filtered_rules = {k: v for k, v in filtered_rules.items() if v.has_all_tags(tags)}
            else:
                filtered_rules = {k: v for k, v in filtered_rules.items() if v.has_any_tags(tags)}

        return filtered_rules

    def detect_duplicates(self, semantic_threshold: float = 0.8,
                         behavioral_threshold: float = 0.9) -> Dict[str, Any]:
        """Detect potential duplicate rules"""
        duplicates = self.analyzer.find_duplicates(
            self.rules, semantic_threshold, behavioral_threshold
        )

        # Group duplicates by type
        semantic_duplicates = [d for d in duplicates if d[3] == "semantic"]
        behavioral_duplicates = [d for d in duplicates if d[3] == "behavioral"]

        return {
            "total_duplicates": len(duplicates),
            "semantic_duplicates": semantic_duplicates,
            "behavioral_duplicates": behavioral_duplicates
        }

    def optimize_execution_order(self) -> List[str]:
        """Optymalizuje kolejność wykonania reguł"""
        # Strategia: najpierw ATOMIC, potem SIMPLE, itd.
        ordered_rules = []

        for level in ["ATOMIC", "SIMPLE", "COMPOUND", "COMPLEX", "COMPOSITE"]:
            level_rules = self.get_rules_by_complexity(level)
            # Sortuj w obrębie poziomu według wydajności
            sorted_level = sorted(
                level_rules.items(),
                key=lambda x: (x[1].execution_cost, -x[1].post_mean)
            )
            ordered_rules.extend([rule_id for rule_id, _ in sorted_level])

        return ordered_rules

    def compose_rules(self, rule_ids: List[str], composite_id: str) -> Optional[Rule]:
        """Tworzy regułę kompozytową z prostszych reguł"""
        if len(rule_ids) < 2:
            return None

        # Sprawdź czy wszystkie reguły istnieją
        rules_to_compose = []
        for rid in rule_ids:
            if rid not in self.rules:
                return None
            rules_to_compose.append(self.rules[rid])

        # Utwórz regułę kompozytową
        composite_rule = Rule(
            id=composite_id,
            name=f"Composite: {', '.join(rule_ids)}",
            rtype=HYBRID,  # Kompozyty są hybrydowe
            complexity_level="COMPOSITE",
            child_rules=rule_ids,
            category="composite",
            tags={"composite", "hierarchical"},
            metadata={"composed_from": rule_ids}
        )

        return composite_rule

    def analyze_rule_patterns(self) -> Dict[str, Any]:
        """Analyze patterns across all rules"""
        if not self.rules:
            return {"error": "no_rules"}

        # Type distribution
        type_counts = defaultdict(int)
        for rule in self.rules.values():
            type_counts[rule.type] += 1

        # Category distribution
        category_counts = defaultdict(int)
        for rule in self.rules.values():
            category_counts[rule.category] += 1

        # Complexity distribution
        complexity_counts = defaultdict(int)
        for rule in self.rules.values():
            complexity_counts[rule.complexity_level] += 1

        # Tag frequency
        tag_counts = defaultdict(int)
        for rule in self.rules.values():
            for tag in rule.tags:
                tag_counts[tag] += 1

        return {
            "total_rules": len(self.rules),
            "type_distribution": dict(type_counts),
            "category_distribution": dict(category_counts),
            "complexity_distribution": dict(complexity_counts),
            "tag_frequency": dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        }

    def get_complexity_stats(self) -> Dict[str, Any]:
        """Zwraca statystyki złożoności systemu"""
        stats = {
            "complexity_distribution": {},
            "average_execution_cost": 0.0,
            "total_rules": len(self.rules),
            "optimization_potential": 0.0
        }

        total_cost = 0.0
        for level in COMPLEXITY_LEVELS.keys():
            level_rules = self.get_rules_by_complexity(level)
            stats["complexity_distribution"][level] = len(level_rules)

            for rule in level_rules.values():
                total_cost += rule.execution_cost

        if self.rules:
            stats["average_execution_cost"] = total_cost / len(self.rules)

            # Potencjał optymalizacji (% reguł ATOMIC i SIMPLE)
            simple_count = (stats["complexity_distribution"].get("ATOMIC", 0) +
                          stats["complexity_distribution"].get("SIMPLE", 0))
            stats["optimization_potential"] = simple_count / len(self.rules)

        return stats

# Dodatkowe funkcje pomocnicze dla v6
def create_physics_rule(rule_id: str, name: str, tolerance: float = 1e-6,
                       priority: int = 10) -> Rule:
    """Helper function to create physics conservation rules"""
    return Rule(
        id=rule_id,
        name=name,
        rtype=LOGICAL,
        category=RULE_CATEGORIES["PHYSICS"],
        tags={"physics", "conservation", "fundamental"},
        priority=priority,
        params={"tolerance": RuleParameter(tolerance, min_val=1e-9, max_val=1e-3)},
        conflict_groups={"conservation_laws"}
    )

def create_optimization_rule(rule_id: str, name: str, alpha: float = 0.5,
                           beta: float = 0.3) -> Rule:
    """Helper function to create optimization rules"""
    return Rule(
        id=rule_id,
        name=name,
        rtype=HYBRID,
        category=RULE_CATEGORIES["OPTIMIZATION"],
        tags={"optimization", "adaptive", "learning"},
        params={
            "alpha": RuleParameter(alpha, min_val=0.0, max_val=1.0, requires_grad=True),
            "beta": RuleParameter(beta, min_val=0.0, max_val=1.0, requires_grad=True)
        }
    )

def create_pattern_rule(rule_id: str, name: str, threshold: float = 0.7) -> Rule:
    """Helper function to create pattern recognition rules"""
    return Rule(
        id=rule_id,
        name=name,
        rtype=HEURISTIC,
        category=RULE_CATEGORIES["PATTERN"],
        tags={"pattern", "recognition", "heuristic"},
        params={"threshold": RuleParameter(threshold, min_val=0.1, max_val=0.9)}
    )

# Funkcja demonstracyjna
def demo_drm_v6():
    """Demonstracja możliwości DRM v6"""
    print("=== DEMO DRM v6 - Zaawansowana Architektura Reguł ===\n")
    
    # Tworzenie systemu
    drm = DRMSystem()
    
    # Dodawanie różnych typów reguł
    rules_to_add = [
        create_physics_rule("mass_conservation", "Mass Conservation Law"),
        create_physics_rule("energy_conservation", "Energy Conservation Law", tolerance=1e-5),
        create_optimization_rule("gradient_descent", "Gradient Descent Optimizer"),
        create_pattern_rule("anomaly_detection", "Anomaly Detection Pattern"),
        create_pattern_rule("trend_recognition", "Trend Recognition Pattern", threshold=0.8)
    ]
    
    for rule in rules_to_add:
        result = drm.add_rule(rule)
        print(f"Dodano regułę {rule.id}: {result['success']}")
    
    print(f"\nLiczba reguł w systemie: {len(drm.rules)}")
    
    # Demonstracja wyszukiwania
    print("\n=== WYSZUKIWANIE I FILTROWANIE ===")
    
    physics_rules = drm.search_rules_by_tags(["physics"])
    print(f"Reguły fizyczne: {list(physics_rules.keys())}")
    
    logical_rules = drm.search_rules_by_type(LOGICAL)
    print(f"Reguły logiczne: {list(logical_rules.keys())}")
    
    # Filtrowanie zaawansowane
    filtered = drm.filter_rules(
        rule_types=[LOGICAL, HYBRID],
        tags=["physics", "optimization"],
        match_all_tags=False
    )
    print(f"Reguły fizyczne lub optymalizacyjne: {list(filtered.keys())}")
    
    # Analiza wzorców
    print("\n=== ANALIZA WZORCÓW ===")
    patterns = drm.analyze_rule_patterns()
    print(f"Rozkład typów: {patterns['type_distribution']}")
    print(f"Rozkład kategorii: {patterns['category_distribution']}")
    print(f"Najczęstsze tagi: {list(patterns['tag_frequency'].keys())[:3]}")
    
    print("\n=== DEMO ZAKOŃCZONE ===")

if __name__ == "__main__":
    demo_drm_v6()
