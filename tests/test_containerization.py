import pytest
from pathlib import Path

def test_docker_configs():
    # Verify file existences
    dockerfile = Path("Dockerfile")
    composefile = Path("docker-compose.yml")
    
    assert dockerfile.exists(), "Dockerfile is missing"
    assert composefile.exists(), "docker-compose.yml is missing"
    
    # Read Dockerfile content
    df_content = dockerfile.read_text(encoding="utf-8")
    assert "FROM python" in df_content
    assert "EXPOSE 8000" in df_content
    assert "uvicorn" in df_content
    
    # Read docker-compose content
    dc_content = composefile.read_text(encoding="utf-8")
    assert "transgraph-backend" in dc_content
    assert "8000:8000" in dc_content
