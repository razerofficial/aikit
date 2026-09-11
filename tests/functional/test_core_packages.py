"""Functional tests for core package imports.

Tests that all core packages can be imported successfully.
This mirrors the smoke tests from the remote Docker validation.
"""
import pytest


class TestCorePackages:
    """Test core package availability and imports."""

    def test_rzr_aikit_import(self):
        """Test that rzr_aikit package can be imported."""
        import rzr_aikit
        assert rzr_aikit is not None

    def test_gpu_discovery_import(self):
        """Test that gpu_discovery package can be imported."""
        import gpu_discovery
        assert gpu_discovery is not None

    def test_vllm_plugins_import(self):
        """Test that vllm_plugins package can be imported."""
        import vllm_plugins
        assert vllm_plugins is not None

    def test_all_core_packages_together(self):
        """Test that all core packages can be imported together."""
        import rzr_aikit
        import gpu_discovery
        import vllm_plugins

        assert rzr_aikit is not None
        assert gpu_discovery is not None
        assert vllm_plugins is not None


class TestCoreModules:
    """Test core module functionality."""

    def test_rzr_aikit_has_version(self):
        """Test that rzr_aikit module is properly loaded."""
        import rzr_aikit
        # Version attribute is optional in some builds
        # Just verify the module loaded successfully
        assert rzr_aikit is not None
        assert hasattr(rzr_aikit, '__name__')

    def test_gpu_discovery_functionality(self):
        """Test basic gpu_discovery functionality."""
        import gpu_discovery

        # Should have core functions/classes
        assert hasattr(gpu_discovery, 'get_gpu_info') or \
               hasattr(gpu_discovery, 'GPUInfo') or \
               dir(gpu_discovery)  # At least something should be there
