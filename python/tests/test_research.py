from pathlib import Path
import sys
import tomllib


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verify_research import DEFAULT_CONFIG, ROOT, audit, parameter_count


def test_canonical_model_exceeds_45_1m_parameters():
    assert parameter_count(vocab_size=1999, d_model=512, n_layers=12, d_ff=2624) == 45_363_968


def test_research_contract_has_no_failures():
    checks = audit(DEFAULT_CONFIG)
    assert not [check for check in checks if check.status == "FAIL"]
    assert {check.name for check in checks if check.status == "PASS"} >= {
        "parameters",
        "architecture",
        "tokenizer",
    }


def test_cafune_mini_configuration_is_a_7m_ablation_model():
    config_path = ROOT / "config" / "experiments" / "cafune-mini.toml"
    model = tomllib.loads(config_path.read_text(encoding="utf-8"))["model"]
    assert parameter_count(
        vocab_size=model["vocab_size"],
        d_model=model["d_model"],
        n_layers=model["n_layers"],
        d_ff=model["d_ff"],
    ) == 7_071_744
