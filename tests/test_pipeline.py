"""Tests for pipeline loader and validator."""

import pytest
import tempfile
from pathlib import Path

from neuralforge.pipeline.loader import PipelineLoader, PipelineLoadError
from neuralforge.pipeline.validator import PipelineValidator, ValidationError
from neuralforge.pipeline.templates import get_template, list_templates, TEMPLATES


class TestPipelineLoader:
    def test_from_dict_basic(self):
        data = {
            "name": "test",
            "nodes": [
                {"id": "a", "agent": "researcher"},
                {"id": "b", "agent": "coder", "depends_on": ["a"]},
            ],
        }
        dag, meta = PipelineLoader.from_dict(data)
        assert meta["name"] == "test"
        assert len(dag.nodes) == 2

    def test_from_yaml_file(self):
        yaml_content = """
name: test_pipeline
version: "1.0"
nodes:
  - id: step1
    agent: researcher
    config:
      query: "test query"
  - id: step2
    agent: coder
    depends_on: [step1]
"""
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            dag, meta = PipelineLoader.from_yaml(f.name)
            assert len(dag.nodes) == 2

    def test_missing_nodes_raises(self):
        with pytest.raises(PipelineLoadError):
            PipelineLoader.from_dict({"name": "empty"})

    def test_nonexistent_file_raises(self):
        with pytest.raises(PipelineLoadError):
            PipelineLoader.from_yaml("/nonexistent/path.yaml")

    def test_roundtrip(self):
        dag, meta = PipelineLoader.from_dict({
            "name": "rt",
            "nodes": [{"id": "a", "agent": "researcher"}],
        })
        yaml_str = PipelineLoader.to_yaml(dag, meta)
        assert "rt" in yaml_str
        assert "researcher" in yaml_str


class TestPipelineValidator:
    def test_valid_pipeline(self):
        errors = PipelineValidator.validate({
            "name": "test",
            "nodes": [{"id": "a", "agent": "researcher"}],
        })
        assert len(errors) == 0

    def test_missing_agent(self):
        errors = PipelineValidator.validate({
            "name": "test",
            "nodes": [{"id": "a"}],
        })
        assert any("agent" in e.message for e in errors)

    def test_duplicate_id(self):
        errors = PipelineValidator.validate({
            "name": "test",
            "nodes": [
                {"id": "a", "agent": "researcher"},
                {"id": "a", "agent": "coder"},
            ],
        })
        assert any("Duplicate" in e.message for e in errors)

    def test_forward_reference(self):
        errors = PipelineValidator.validate({
            "name": "test",
            "nodes": [
                {"id": "b", "agent": "coder", "depends_on": ["nonexistent"]},
            ],
        })
        assert any("not found" in e.message for e in errors)


class TestTemplates:
    def test_list_templates(self):
        templates = list_templates()
        assert len(templates) > 0
        assert "research_and_code" in templates

    def test_get_template(self):
        tmpl = get_template("research_and_code")
        assert "nodes" in tmpl
        assert len(tmpl["nodes"]) >= 2

    def test_get_nonexistent(self):
        with pytest.raises(KeyError):
            get_template("nonexistent")

    def test_all_templates_valid(self):
        for name in list_templates():
            tmpl = get_template(name)
            dag, _ = PipelineLoader.from_dict(tmpl)  # Should not raise
            assert len(dag.nodes) > 0
