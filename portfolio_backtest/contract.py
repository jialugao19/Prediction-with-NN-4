"""Define stable input and output contracts for the portfolio backtest package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


INFERENCE_MANIFEST_COLUMNS = ["prediction", "code", "date", "time"]
POSITION_TABLE_COLUMNS = [
    "date",
    "time",
    "minute_slot",
    "slot_bar_id",
    "code",
    "prediction",
    "prediction_available",
    "current_tradable",
    "simple_return",
    "sigma_intraday",
    "adv_amount",
    "fillable",
    "spread_bps",
    "side",
    "target_weight",
]
BAR_OUTPUT_COLUMNS = [
    "minute_slot",
    "slot_bar_id",
    "date",
    "time",
    "gross_return",
    "planned_turnover",
    "turnover",
    "spread_cost",
    "impact_coeff",
    "planned_name_count",
    "filled_name_count",
    "fill_ratio",
    "long_exposure",
    "short_exposure",
    "executed_gross_exposure",
    "cash_buffer",
]


@dataclass(frozen=True)
class PortfolioBacktestConfig:
    """Store the fixed IO layout and portfolio backtest assumptions."""

    repo_root: Path
    inference_manifest_path: Path
    output_dir: Path
    feature_db_path: Path
    feature_chunk_dir: Path
    feature_manifest_path: Path
    raw_stock1m_root: Path
    raw_stock1d_root: Path
    stock_basic_path: Path
    namechange_path: Path
    top_frac: float
    slot_mod_bars: int
    long_enabled: bool
    short_enabled: bool
    max_liq_bucket: int
    entry_delay_bars: int
    holding_bars: int
    annual_days: int
    lookup_cache_size: int
    adv_lookback_days: int
    sigma_lookback_bars: int
    impact_eta: float
    spread_bps_high: float
    spread_bps_mid: float
    spread_bps_low: float
    aum_list: list[float]
    impact_budget_bps_list: list[float]
    report_title: str


@dataclass(frozen=True)
class PortfolioBacktestInputContract:
    """Describe the required inputs for one portfolio backtest run."""

    version: str
    manifest_type: str
    required_manifest_columns: list[str]
    expected_manifest_filename: str
    expected_inference_split: str
    required_position_columns: list[str]


@dataclass(frozen=True)
class PortfolioBacktestOutputContract:
    """Describe the stable artifacts emitted by one portfolio backtest run."""

    version: str
    output_dir_name: str
    required_artifacts: list[str]
    slot_bar_columns: list[str]


@dataclass(frozen=True)
class PortfolioBacktestRuntimeContract:
    """Bundle the input and output contracts written with each run."""

    input_contract: PortfolioBacktestInputContract
    output_contract: PortfolioBacktestOutputContract

    def to_dict(self) -> dict[str, object]:
        """Convert the runtime contract into a YAML-friendly dict."""
        # Materialize nested dataclasses as plain Python containers.
        return {
            "input_contract": {
                "version": str(self.input_contract.version),
                "manifest_type": str(self.input_contract.manifest_type),
                "required_manifest_columns": list(self.input_contract.required_manifest_columns),
                "expected_manifest_filename": str(self.input_contract.expected_manifest_filename),
                "expected_inference_split": str(self.input_contract.expected_inference_split),
                "required_position_columns": list(self.input_contract.required_position_columns),
            },
            "output_contract": {
                "version": str(self.output_contract.version),
                "output_dir_name": str(self.output_contract.output_dir_name),
                "required_artifacts": list(self.output_contract.required_artifacts),
                "slot_bar_columns": list(self.output_contract.slot_bar_columns),
            },
        }


def build_default_portfolio_backtest_config() -> PortfolioBacktestConfig:
    """Build the canonical portfolio backtest configuration."""
    # Define the fixed repo paths and canonical inference input.
    repo_root = Path("/home/maomao/prediction-NN-2")
    inference_manifest_path = Path("/data-cache/nn/0428/date_ranges/run/inference_test/iter_140000/inference_manifest.yaml")

    # Define the portfolio backtest output layout.
    output_dir = Path("/data-cache/nn/0426/portfolio_backtest")
    feature_db_path = output_dir / "portfolio_backtest.duckdb"
    feature_chunk_dir = output_dir / "feature_chunks"
    feature_manifest_path = output_dir / "feature_manifest.yaml"

    # Define the market data inputs.
    raw_stock1m_root = Path("/data/ashare/market/stock1m")
    raw_stock1d_root = Path("/data/ashare/market/stock1d")
    stock_basic_path = Path("/data/ashare/market/stock_basic.csv")
    namechange_path = Path("/data/ashare/market/namechange.csv")

    # Define the selection, horizon, and reporting knobs.
    top_frac = 0.10
    slot_mod_bars = 10
    long_enabled = True
    short_enabled = True
    max_liq_bucket = 3
    entry_delay_bars = 1
    holding_bars = 10
    annual_days = 252
    lookup_cache_size = 6
    adv_lookback_days = 20
    sigma_lookback_bars = 20

    # Define the cost model knobs.
    impact_eta = 0.50
    spread_bps_high = 5.0
    spread_bps_mid = 10.0
    spread_bps_low = 20.0
    aum_list = [10_000_000.0, 50_000_000.0, 100_000_000.0]
    impact_budget_bps_list = [10.0, 20.0]
    report_title = "Portfolio Backtest: Execution-Aware Strategy Backtest"
    return PortfolioBacktestConfig(
        repo_root=repo_root,
        inference_manifest_path=inference_manifest_path,
        output_dir=output_dir,
        feature_db_path=feature_db_path,
        feature_chunk_dir=feature_chunk_dir,
        feature_manifest_path=feature_manifest_path,
        raw_stock1m_root=raw_stock1m_root,
        raw_stock1d_root=raw_stock1d_root,
        stock_basic_path=stock_basic_path,
        namechange_path=namechange_path,
        top_frac=top_frac,
        slot_mod_bars=slot_mod_bars,
        long_enabled=long_enabled,
        short_enabled=short_enabled,
        max_liq_bucket=max_liq_bucket,
        entry_delay_bars=entry_delay_bars,
        holding_bars=holding_bars,
        annual_days=annual_days,
        lookup_cache_size=lookup_cache_size,
        adv_lookback_days=adv_lookback_days,
        sigma_lookback_bars=sigma_lookback_bars,
        impact_eta=impact_eta,
        spread_bps_high=spread_bps_high,
        spread_bps_mid=spread_bps_mid,
        spread_bps_low=spread_bps_low,
        aum_list=aum_list,
        impact_budget_bps_list=impact_budget_bps_list,
        report_title=report_title,
    )


def ensure_output_dir(path: Path) -> None:
    """Create one output directory."""
    # Create the directory tree.
    path.mkdir(parents=True, exist_ok=True)


def build_input_contract() -> PortfolioBacktestInputContract:
    """Build the stable input contract used by the package."""
    # Pin the inference manifest and position-table requirements.
    return PortfolioBacktestInputContract(
        version="v1",
        manifest_type="inference_manifest",
        required_manifest_columns=list(INFERENCE_MANIFEST_COLUMNS),
        expected_manifest_filename="inference_manifest.yaml",
        expected_inference_split="inference_test",
        required_position_columns=list(POSITION_TABLE_COLUMNS),
    )


def build_output_contract(output_dir: Path) -> PortfolioBacktestOutputContract:
    """Build the stable output contract used by the package."""
    # Pin the artifact set emitted by the orchestration entrypoint.
    return PortfolioBacktestOutputContract(
        version="v1",
        output_dir_name=str(Path(output_dir).name),
        required_artifacts=[
            "feature_audit.csv",
            "feature_audit.yaml",
            "feature_manifest.yaml",
            "open_slot_positions.parquet",
            "vwap_slot_positions.parquet",
            "baseline_open_slot_bar.csv",
            "baseline_open_combined_daily.csv",
            "baseline_open_slot_summary.csv",
            "realistic_vwap_slot_bar.csv",
            "realistic_vwap_combined_daily.csv",
            "realistic_vwap_slot_summary.csv",
            "strategy_summary.yaml",
            "research_report.md",
            "research_report.html",
            "strategy_curves.png",
            "baseline_open_strategy.png",
            "drawdown_curve.png",
            "slot_sharpe.png",
            "capacity_sweep.png",
            "portfolio_backtest_contract.yaml",
        ],
        slot_bar_columns=list(BAR_OUTPUT_COLUMNS),
    )


def write_runtime_contract(output_dir: Path) -> Path:
    """Write the stable portfolio backtest runtime contract as YAML."""
    # Build the nested contract payload once.
    runtime_contract = PortfolioBacktestRuntimeContract(
        input_contract=build_input_contract(),
        output_contract=build_output_contract(output_dir),
    )

    # Persist the YAML payload inside the output directory.
    ensure_output_dir(Path(output_dir))
    contract_path = Path(output_dir) / "portfolio_backtest_contract.yaml"
    contract_path.write_text(yaml.safe_dump(runtime_contract.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")
    return contract_path


def read_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML file into a dict."""
    # Load the payload exactly once.
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_inference_manifest(manifest_path: Path) -> dict[str, Any]:
    """Read and validate one inference manifest file."""
    # Load the manifest payload and the expected input contract.
    manifest = read_yaml(manifest_path)
    contract = build_input_contract()
    columns = [str(col) for col in list(manifest["columns"])]

    # Enforce the exact inference manifest schema and naming contract.
    if Path(manifest_path).name != str(contract.expected_manifest_filename):
        raise RuntimeError(f"Unexpected inference manifest filename: {manifest_path}")
    if columns != list(contract.required_manifest_columns):
        raise RuntimeError(f"Unexpected inference manifest columns: {columns}")
    if str(Path(manifest_path).parent.parent.name) != str(contract.expected_inference_split):
        raise RuntimeError(f"Unexpected inference split directory: {manifest_path}")
    return manifest


def load_manifest_glob(manifest_path: Path) -> str:
    """Resolve the parquet glob from one validated manifest YAML."""
    # Read the manifest once and recover the chunk folder path.
    manifest = load_inference_manifest(manifest_path)
    manifest_dir = Path(manifest_path).parent
    first_chunk_parent = Path(manifest["chunk_files"][0]).parent.as_posix()
    return (manifest_dir / first_chunk_parent / "*.parquet").as_posix()


def load_manifest_chunk_paths(manifest_path: Path) -> list[Path]:
    """Resolve concrete parquet chunk paths from one validated manifest YAML."""
    # Read the manifest once and resolve each relative chunk file.
    manifest = load_inference_manifest(manifest_path)
    manifest_dir = Path(manifest_path).parent
    return [manifest_dir / Path(chunk_file) for chunk_file in manifest["chunk_files"]]


def load_chunk_manifest(manifest_path: Path) -> dict[str, Any]:
    """Read one generic parquet chunk manifest without inference-schema validation."""
    # Load the manifest payload once.
    return read_yaml(manifest_path)


def load_chunk_manifest_glob(manifest_path: Path) -> str:
    """Resolve the parquet glob from one generic chunk manifest YAML."""
    # Read the manifest once and recover the chunk folder path.
    manifest = load_chunk_manifest(manifest_path)
    manifest_dir = Path(manifest_path).parent
    first_chunk_parent = Path(manifest["chunk_files"][0]).parent.as_posix()
    return (manifest_dir / first_chunk_parent / "*.parquet").as_posix()


def load_chunk_manifest_paths(manifest_path: Path) -> list[Path]:
    """Resolve concrete parquet chunk paths from one generic chunk manifest YAML."""
    # Read the manifest once and resolve each relative chunk file.
    manifest = load_chunk_manifest(manifest_path)
    manifest_dir = Path(manifest_path).parent
    return [manifest_dir / Path(chunk_file) for chunk_file in manifest["chunk_files"]]


def write_chunk_manifest(manifest_path: Path, chunk_paths: list[Path], chunk_dir_name: str, columns: list[str]) -> None:
    """Write one parquet-chunk manifest YAML."""
    # Build relative chunk filenames and serialize as YAML.
    chunk_files = [str(Path(chunk_dir_name) / path.name) for path in list(chunk_paths)]
    payload = {"format": "parquet_chunks", "columns": list(columns), "chunk_files": list(chunk_files)}
    Path(manifest_path).write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def validate_required_artifacts(output_dir: Path) -> None:
    """Validate that one output directory satisfies the output contract."""
    # Require every contract artifact to exist after orchestration finishes.
    output_dir = Path(output_dir)
    contract = build_output_contract(output_dir)
    for artifact_name in list(contract.required_artifacts):
        artifact_path = output_dir / str(artifact_name)
        if not artifact_path.exists():
            raise RuntimeError(f"Missing portfolio backtest artifact: {artifact_path}")
