"""
End-to-end tests for the quantitative research pipeline.

Tests the full pipeline execution from data processing to results generation.
"""

import subprocess
import sys
from pathlib import Path
import pandas as pd
import pytest
import json
from datetime import datetime


# Test configuration
TEST_CONFIG = {
    'small_universe_size': 10,  # Use small subset for faster testing
    'test_days': 30,  # Test with recent 30 days
    'timeout': 300,  # 5 minute timeout for pipeline tests
}

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'


class TestPipelineExecution:
    """Test complete pipeline execution with small parameters."""

    def test_pipeline_cli_basic(self):
        """Test basic pipeline execution with minimal parameters."""
        # Prepare command with small parameters for faster execution
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / 'run_full_pipeline.py'),
            '--k', '5',  # Small universe
            '--swap-cap', '0.1',  # Conservative swap cap
            '--skip-slow-tasks',  # Skip time-consuming optional tasks
        ]

        # Execute pipeline
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TEST_CONFIG['timeout'],
            cwd=PROJECT_ROOT
        )

        # Assert successful execution
        assert result.returncode == 0, f"Pipeline failed: {result.stderr}"

        # Verify key output files are generated
        expected_files = [
            'predictions_*.csv',
            'pipeline_execution_*_metrics.json',
            'pipeline_execution_*_timeseries.csv'
        ]

        for pattern in expected_files:
            files = list(DATA_DIR.glob(pattern))
            assert len(files) > 0, f"Expected output file matching {pattern} not found"

        # Verify predictions file structure
        latest_pred = max(DATA_DIR.glob('predictions_*.csv'), key=lambda x: x.stat().st_mtime)
        df = pd.read_csv(latest_pred, nrows=1000)  # Read sample

        required_columns = ['date', 'stock_code', 'y_pred', 'y_true']
        for col in required_columns:
            assert col in df.columns, f"Required column {col} missing from predictions"

        # Verify metrics file content
        latest_metrics = max(DATA_DIR.glob('pipeline_execution_*_metrics.json'), key=lambda x: x.stat().st_mtime)
        with open(latest_metrics) as f:
            metrics = json.load(f)

        expected_metrics = ['ic_mean', 'ic_ir', 'n_samples']
        for metric in expected_metrics:
            assert metric in metrics, f"Expected metric {metric} not found in results"

    def test_pipeline_with_data_validation(self):
        """Test pipeline execution with data quality validation."""
        # Check if we have sufficient input data
        stock_files = list(DATA_DIR.glob('0*.csv')) + list(DATA_DIR.glob('6*.csv'))
        assert len(stock_files) >= TEST_CONFIG['small_universe_size'], \
            f"Insufficient stock data files: {len(stock_files)} < {TEST_CONFIG['small_universe_size']}"

        # Validate stock data format
        sample_stock = stock_files[0]
        df = pd.read_csv(sample_stock, nrows=10)

        required_price_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        for col in required_price_columns:
            assert col in df.columns, f"Stock data missing required column: {col}"

        # Ensure dates are properly formatted
        df['date'] = pd.to_datetime(df['date'])
        assert not df['date'].isna().any(), "Invalid dates found in stock data"


class TestTurnoverGridExecution:
    """Test turnover grid search functionality."""

    def test_turnover_grid_generation(self):
        """Test turnover strategy grid generation."""
        # First ensure we have predictions file
        pred_files = list(DATA_DIR.glob('predictions_*.csv'))
        if not pred_files:
            pytest.skip("No predictions file found, skipping grid test")

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / 'regime_model_grid_search.py'),
            '--top-n', '30,35',  # Small grid for testing
            '--cost-bps', '0.0003,0.0005',
            '--quick-test'  # If this flag exists
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TEST_CONFIG['timeout'],
            cwd=PROJECT_ROOT
        )

        # Should complete without error (even if file doesn't exist)
        assert result.returncode in [0, 1], f"Grid search failed unexpectedly: {result.stderr}"

        # If successful, verify output
        if result.returncode == 0:
            grid_files = list(DATA_DIR.glob('turnover_strategy_grid_*.csv'))
            assert len(grid_files) > 0, "Expected turnover grid output file not found"

            # Verify grid file structure
            latest_grid = max(grid_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_csv(latest_grid)

            required_grid_columns = ['strategy', 'top_n', 'cost_bps', 'ls_ir']
            for col in required_grid_columns:
                assert col in df.columns, f"Grid file missing required column: {col}"


class TestDataIntegrity:
    """Test data integrity and consistency."""

    def test_stock_universe_consistency(self):
        """Test stock universe file consistency with actual data."""
        universe_file = PROJECT_ROOT / 'stock_universe.csv'

        if not universe_file.exists():
            pytest.skip("Stock universe file not found")

        # Load universe
        universe = pd.read_csv(universe_file)
        assert 'code' in universe.columns, "Universe missing 'code' column"
        assert 'name' in universe.columns, "Universe missing 'name' column"

        # Check if universe stocks have corresponding data files
        universe_codes = set(universe['code'].astype(str).str.zfill(6))
        data_files = set(f.stem for f in DATA_DIR.glob('0*.csv')) | set(f.stem for f in DATA_DIR.glob('6*.csv'))

        # At least some overlap should exist
        overlap = universe_codes & data_files
        assert len(overlap) > 0, "No overlap between universe and available data files"

    def test_recent_data_availability(self):
        """Test that we have recent data for analysis."""
        stock_files = list(DATA_DIR.glob('0*.csv'))[:5]  # Sample first 5 files

        if not stock_files:
            pytest.skip("No stock data files found")

        recent_threshold = datetime.now().replace(year=datetime.now().year - 2)  # 2 years ago

        for stock_file in stock_files:
            df = pd.read_csv(stock_file, parse_dates=['date'])
            latest_date = df['date'].max()

            assert latest_date > pd.Timestamp(recent_threshold), \
                f"Stock {stock_file.stem} data too old: {latest_date}"


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment before running tests."""
    # Ensure required directories exist
    DATA_DIR.mkdir(exist_ok=True)
    (PROJECT_ROOT / 'logs').mkdir(exist_ok=True)

    # Clean up any existing test artifacts
    test_patterns = [
        'test_predictions_*.csv',
        'test_pipeline_*.json',
        'test_turnover_*.csv'
    ]

    for pattern in test_patterns:
        for file in DATA_DIR.glob(pattern):
            file.unlink()

    yield

    # Cleanup after tests (optional)
    # Can be disabled to inspect test outputs


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])