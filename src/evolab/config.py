"""Configuration models for EvolutionEngine and modular GA components."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SpeciationConfig:
    """Configuration for speciation distance and threshold dynamics."""
    enabled: bool = True
    threshold: float = 0.65
    c1: float = 0.6
    c2: float = 0.4
    c3: float = 0.0
    metric: str = "composite"  # "composite", "euclidean", "maxdelta"


@dataclass
class QualityDiversityConfig:
    """Configuration for MAP-Elites behavioral archive."""
    enabled: bool = True
    grid_x: int = 8
    grid_y: int = 6
    k: int = 16  # Target archive resolution / capacity
    scale_x: float = 10.0
    scale_y: float = 2.5
    active_selection: bool = False  # If True, sample parents from diverse archive cells


@dataclass
class MemoryConfig:
    """Configuration for temporal memory and environmental change detection."""
    enabled: bool = False
    max_size: int = 1000  # Max capacity for solution memory buffer
    max_injection_rate: float = 0.15
    change_window: int = 20
    cusum_k: float = 0.5
    cusum_h: float = 5.0
    staleness_tau: float = 100.0
    quarantine_gens: int = 5


@dataclass
class EngineConfig:
    """Comprehensive, clean configuration for EvolutionEngine."""
    # Core population & budget parameters
    population_size: int = 16
    generations: int = 100
    elite_count: int = 2
    mutation_rate: float = 0.15
    mutation_boost: float = 1.0
    early_stop_fitness: float | None = None
    seed: int | None = None
    genome_size: int = 16
    stagnation_patience: int = 15
    immigrant_fraction: float = 0.0
    fitness_range: tuple[float, float] = (0.0, 100.0)
    allow_1d: bool = False

    # Fitness sharing & scheduling
    fitness_sharing: bool = True
    sharing_mode: str = "dynamic"  # "dynamic", "static", "off"
    exploit_after_frac: float = 2.0 / 3.0
    hybrid_light_share: float = 0.7
    mutation_enabled: bool = True
    crossover_rate: float = 0.8
    local_search_steps: int = 0
    crossover_mode: str = "single_point"  # "single_point", "blend", "uniform"

    # Sub-component configurations
    speciation: SpeciationConfig = field(default_factory=SpeciationConfig)
    qd: QualityDiversityConfig = field(default_factory=QualityDiversityConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    # Evaluation & constraints
    eval_repeats: int = 1
    stability_penalty: float = 0.0
    hard_constraints: Sequence[Callable] = field(default_factory=tuple)
    record_population_snapshots: bool = False
    record_archive_solutions: bool = False


# ==============================================================================
# Declarative Project Configuration (evolab.toml / .evolab.json / pyproject.toml)
# ==============================================================================

def find_project_config_file(start_dir: Path | str | None = None) -> Path | None:
    """Search upwards from start_dir for evolab.toml, .evolab.json, or pyproject.toml."""
    cur = Path(start_dir or ".").resolve()
    for directory in [cur, *cur.parents]:
        for candidate in ["evolab.toml", ".evolab.json", "pyproject.toml"]:
            target = directory / candidate
            if target.is_file():
                if candidate == "pyproject.toml":
                    try:
                        content = target.read_text(encoding="utf-8")
                        if "[tool.evolab]" in content:
                            return target
                    except Exception:
                        continue
                else:
                    return target
    return None


def find_user_config_file() -> Path | None:
    """Search for global user configuration in ~/.evolab/config.toml, ~/.darwinrc, or XDG config."""
    home = Path.home()
    candidates = [
        home / ".evolab" / "config.toml",
        home / ".evolab.toml",
        home / ".darwinrc",
        home / ".config" / "evolab" / "config.toml",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _parse_config_file(cfg_file: Path) -> dict[str, Any]:
    """Parses a TOML or JSON config file and returns a flattened dictionary."""
    try:
        content = cfg_file.read_text(encoding="utf-8")
    except Exception:
        return {}

    raw_data: dict[str, Any] = {}
    if cfg_file.suffix == ".json":
        import json
        try:
            raw_data = json.loads(content)
        except Exception:
            return {}
    else:
        # TOML format
        try:
            import tomllib
            raw_data = tomllib.loads(content)
        except ImportError:
            try:
                import tomli as tomllib
                raw_data = tomllib.loads(content)
            except ImportError:
                raw_data = {}
                current_section = ""
                for line in content.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("[") and line.endswith("]"):
                        current_section = line[1:-1].strip()
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        target_sec = raw_data.setdefault(current_section, {}) if current_section else raw_data
                        if v.lower() in ("true", "false"):
                            target_sec[k] = v.lower() == "true"
                        elif v.isdigit():
                            target_sec[k] = int(v)
                        else:
                            try:
                                target_sec[k] = float(v)
                            except ValueError:
                                target_sec[k] = v

    if "tool" in raw_data and "evolab" in raw_data["tool"]:
        raw_data = raw_data["tool"]["evolab"]

    flat: dict[str, Any] = {}
    for section_key, section_val in raw_data.items():
        if isinstance(section_val, dict):
            for k, v in section_val.items():
                flat[k] = v
        else:
            flat[section_key] = section_val

    if "sources" in flat and "source" not in flat:
        flat["source"] = flat.pop("sources")
    if "tests" in flat and isinstance(flat["tests"], list):
        import json
        flat["tests"] = json.dumps(flat["tests"])

    return flat


def load_project_config(start_dir: Path | str | None = None) -> dict[str, Any]:
    """Load declarative evolab configuration if found in current project directory tree."""
    cfg_file = find_project_config_file(start_dir)
    if not cfg_file:
        return {}
    return _parse_config_file(cfg_file)


# Global provenance tracker
_LAST_CONFIG_PROVENANCE: dict[str, tuple[Any, str]] = {}


def get_config_provenance() -> dict[str, tuple[Any, str]]:
    """Returns provenance of the most recently loaded hierarchical configuration."""
    return dict(_LAST_CONFIG_PROVENANCE)


def _get_env_config(prefix: str = "EVOLAB_") -> dict[str, Any]:
    """Extracts configuration settings from environment variables starting with prefix."""
    import os
    env_map: dict[str, Any] = {}
    prefix_len = len(prefix)

    var_to_key = {
        "POPULATION": "population",
        "POPULATION_SIZE": "population",
        "GENERATIONS": "generations",
        "SEED": "seed",
        "ENGINE": "engine",
        "MUTATION_RATE": "mutation_rate",
        "TARGET": "target",
        "DIFF": "diff",
        "OUTPUT": "output",
        "SCENARIO": "scenario",
        "SOURCE": "source",
        "PYTEST": "pytest",
        "VERBOSE": "verbose",
        "CHECKPOINT_EVERY": "checkpoint_every",
    }

    for k, v in os.environ.items():
        if k.startswith(prefix):
            suffix = k[prefix_len:].upper()
            target_key = var_to_key.get(suffix, suffix.lower())
            val: Any = v
            if v.lower() in ("true", "1", "yes"):
                val = True
            elif v.lower() in ("false", "0", "no"):
                val = False
            elif v.isdigit():
                val = int(v)
            else:
                try:
                    val = float(v)
                except ValueError:
                    val = v
            env_map[target_key] = val

    return env_map


def load_hierarchical_config(
    start_dir: Path | str | None = None,
    cli_overrides: dict[str, Any] | None = None,
    env_prefix: str = "EVOLAB_",
) -> dict[str, Any]:
    """
    Cascading hierarchical configuration loader adhering to UNIX precedence:
    Defaults < User (~/.evolab/config.toml) < Project (evolab.toml) < Environment (EVOLAB_*) < CLI
    """
    global _LAST_CONFIG_PROVENANCE
    provenance: dict[str, tuple[Any, str]] = {}
    merged: dict[str, Any] = {}

    # Layer 1: Code base defaults
    base_defaults = {
        "engine": "auto",
        "generations": 30,
        "population": 16,
        "target": 99.7,
        "seed": None,
        "diff": False,
    }
    for k, v in base_defaults.items():
        merged[k] = v
        provenance[k] = (v, "default")

    # Layer 2: User config
    user_file = find_user_config_file()
    if user_file:
        user_cfg = _parse_config_file(user_file)
        for k, v in user_cfg.items():
            merged[k] = v
            provenance[k] = (v, f"user:{user_file.name}")

    # Layer 3: Project config
    proj_cfg = load_project_config(start_dir)
    if proj_cfg:
        proj_file = find_project_config_file(start_dir)
        proj_name = proj_file.name if proj_file else "project"
        for k, v in proj_cfg.items():
            merged[k] = v
            provenance[k] = (v, f"project:{proj_name}")

    # Layer 4: Environment variables
    env_cfg = _get_env_config(env_prefix)
    for k, v in env_cfg.items():
        merged[k] = v
        provenance[k] = (v, f"env:{env_prefix}{k.upper()}")

    # Layer 5: CLI overrides
    if cli_overrides:
        for k, v in cli_overrides.items():
            if v is not None:
                merged[k] = v
                provenance[k] = (v, "cli")

    _LAST_CONFIG_PROVENANCE = provenance
    return merged



def generate_default_config(fmt: str = "toml") -> str:
    """Generate starter configuration template for evolab init."""
    if fmt.lower() == "json":
        import json
        return json.dumps({
            "project": {
                "sources": ["app.py"],
                "pytest": "tests/test_app.py",
            },
            "engine": {
                "engine": "auto",
                "generations": 30,
                "population": 16,
                "target": 99.7,
                "seed": 42,
            },
            "reporting": {
                "diff": True,
                "output": "run_report.json",
            },
        }, indent=2) + "\n"

    return """# evolab.toml — Project configuration for darwin-evolab

[project]
# Target source file(s) to evolve / repair
sources = ["app.py"]

# Pytest assertion file to evaluate solutions
pytest = "tests/test_app.py"

[engine]
# Search engine: "auto", "greedy" (code), "ga" (numerical), or "nsga2"
engine = "auto"
generations = 30
population = 16
target = 99.7
seed = 42

[reporting]
diff = true
output = "run_report.json"
"""

