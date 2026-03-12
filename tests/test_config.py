from llm_benchmark_suite.config import load_run_config


def test_load_local_demo_config() -> None:
    config = load_run_config("configs/profiles/local-demo.yaml")
    assert config.profile_name == "local-demo"
    assert "vllm" in config.selected_backends
