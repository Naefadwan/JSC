"""Environment-backed application settings with local-only defaults."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from JSC-prefixed environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="JSC_",
        extra="ignore",
    )

    env: Literal["local", "test", "commission"] = "local"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    findings_path: Path = Path("Data/reports/reviews/findings.json")
    reviewer_decisions_path: Path = Path("data/reviewer/decisions.json")
    reports_dir: Path = Path("Data/reports")
    extracted_dir: Path = Path("Data/reports/extracted")
    ocr_dir: Path = Path("Data/reports/ocr")
    requirements_register_path: Path = Path("Data/checklists/requirements_register.csv")
    bm25_index_path: Path = Path("indexes/chunk_bm25.json")
    dense_index_path: Path = Path("indexes/chunk_dense.faiss")
    update_dense_on_upload: bool = True
    # When True, the per-upload review loop also queries the dense FAISS index
    # alongside BM25 (reciprocal rank fusion). Default stays False so an older
    # install without the retrieval extra keeps working unchanged.
    use_dense_at_review: bool = False
    # On-disk location for the golden-label store (two-reviewer agreement set).
    golden_labels_path: Path = Path("data/golden/labels.json")
    max_upload_mb: int = Field(default=100, ge=1, le=1024)
    cer_measurements_path: Path = Path("Data/reports/ocr_rerun/cer_measurements.json")
    chunks_with_ocr_dir: Path = Path("Data/reports/chunks_with_ocr")
    brief_path: Path = Path("Data/owner_review/brief.md")
    auto_refresh_brief: bool = True
    retrieval_gate_measurements_path: Path = Path("Data/reports/retrieval_gate/measurements.json")

    # Company registry (parsed from the local Excel) and the persistent map
    # from an uploaded document to its company.
    companies_excel_path: Path = Path("Data/اسماء الشركات التي نريد تدقيقها.xlsx")
    companies_registry_path: Path = Path("Data/companies/companies.json")
    document_companies_path: Path = Path("Data/documents/document_companies.json")
    disclosure_periods_path: Path = Path("Data/compliance/disclosure_periods.json")

    # Auto-link policy. The pipeline runs the deterministic detector against
    # the first few extracted pages. Links are written only when the top
    # candidate exceeds the threshold AND leads the runner-up by a margin.
    auto_link_company: bool = True
    auto_link_confidence_threshold: float = 0.70
    auto_link_scan_pages: int = 5
    auto_link_min_lead_margin: float = 0.25

    # The cross-encoder reranker is opt-in and off by default.
    #
    # On the current labeled set the measured retrieval gate shows that
    # ``fused+dedup+augment`` reaches 87.5% Hit@5, while adding the reranker
    # drops the same metric to 82.1%. Running the reranker therefore both
    # slows every upload and lowers top-k recall, so production stays with
    # the lighter pipeline. Turning this flag back on reloads the local
    # cross-encoder once per process via ``shared_cross_encoder()``.
    use_reranker: bool = False


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide validated settings."""

    return Settings()
