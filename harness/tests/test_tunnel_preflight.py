"""
Tests for tunnel_preflight module (v0.6.1 SSH tunnel pre-flight checks).

Tests cover:
- Host detection for 8xx series
- Tunnel port mapping
- Socket-based tunnel availability checks
- Tunnel startup via subprocess
- ensure_tunnel_connectivity integration
"""

from unittest.mock import MagicMock, patch

from harness.tunnel_preflight import (
    TUNNEL_PORTS,
    TUNNELED_HOSTS,
    check_tunnel_available,
    ensure_tunnel_connectivity,
    get_tunnel_port,
    get_tunnel_status,
    is_tunneled_host,
    start_tunnel,
    stop_tunnel,
)


class TestTunneledHostDetection:
    """Tests for is_tunneled_host function."""

    def test_vmnode851_is_tunneled(self):
        """vmnode851 should require tunneling."""
        assert is_tunneled_host("vmnode851") is True

    def test_vmnode852_is_tunneled(self):
        """vmnode852 should require tunneling."""
        assert is_tunneled_host("vmnode852") is True

    def test_vmnode876_is_tunneled(self):
        """vmnode876 should require tunneling."""
        assert is_tunneled_host("vmnode876") is True

    def test_vmnode520_is_not_tunneled(self):
        """vmnode520 (5xx series) should NOT require tunneling."""
        assert is_tunneled_host("vmnode520") is False

    def test_vmnode538_is_not_tunneled(self):
        """vmnode538 (production 5xx) should NOT require tunneling."""
        assert is_tunneled_host("vmnode538") is False

    def test_unknown_host_is_not_tunneled(self):
        """Unknown hosts should NOT require tunneling."""
        assert is_tunneled_host("someotherhost") is False

    def test_empty_string_is_not_tunneled(self):
        """Empty string should NOT require tunneling."""
        assert is_tunneled_host("") is False


class TestTunnelPortMapping:
    """Tests for get_tunnel_port function."""

    def test_vmnode851_port(self):
        """vmnode851 should map to port 15851."""
        assert get_tunnel_port("vmnode851") == 15851

    def test_vmnode852_port(self):
        """vmnode852 should map to port 15852."""
        assert get_tunnel_port("vmnode852") == 15852

    def test_vmnode876_port(self):
        """vmnode876 should map to port 15876."""
        assert get_tunnel_port("vmnode876") == 15876

    def test_non_tunneled_host_returns_none(self):
        """Non-tunneled hosts should return None."""
        assert get_tunnel_port("vmnode520") is None
        assert get_tunnel_port("unknown") is None

    def test_constants_match(self):
        """TUNNEL_PORTS and TUNNELED_HOSTS should be consistent."""
        assert set(TUNNEL_PORTS.keys()) == TUNNELED_HOSTS


class TestCheckTunnelAvailable:
    """Tests for check_tunnel_available function."""

    def test_non_tunneled_host_returns_false(self):
        """Non-tunneled hosts should return False immediately."""
        assert check_tunnel_available("vmnode520") is False

    @patch("socket.socket")
    def test_tunnel_available_when_port_open(self, mock_socket_class):
        """Should return True when port is open."""
        mock_sock = MagicMock()
        mock_sock.connect_ex.return_value = 0  # Success
        mock_socket_class.return_value.__enter__.return_value = mock_sock

        assert check_tunnel_available("vmnode852") is True
        mock_sock.connect_ex.assert_called_once_with(("localhost", 15852))

    @patch("socket.socket")
    def test_tunnel_unavailable_when_port_closed(self, mock_socket_class):
        """Should return False when port is closed."""
        mock_sock = MagicMock()
        mock_sock.connect_ex.return_value = 111  # Connection refused
        mock_socket_class.return_value.__enter__.return_value = mock_sock

        assert check_tunnel_available("vmnode852") is False

    @patch("socket.socket")
    def test_tunnel_unavailable_on_socket_error(self, mock_socket_class):
        """Should return False on socket error."""

        mock_socket_class.return_value.__enter__.side_effect = OSError("Connection error")

        assert check_tunnel_available("vmnode876") is False

    @patch("socket.socket")
    def test_timeout_is_set(self, mock_socket_class):
        """Should set socket timeout."""
        mock_sock = MagicMock()
        mock_sock.connect_ex.return_value = 0
        mock_socket_class.return_value.__enter__.return_value = mock_sock

        check_tunnel_available("vmnode851", timeout=2.5)
        mock_sock.settimeout.assert_called_once_with(2.5)


class TestStartTunnel:
    """Tests for start_tunnel function."""

    def test_non_tunneled_host_fails(self, tmp_path):
        """Non-tunneled hosts should fail immediately."""
        success, message = start_tunnel("vmnode520", tmp_path)
        assert success is False
        assert "not a tunneled host" in message

    def test_missing_script_fails(self, tmp_path):
        """Missing tunnel script should fail."""
        success, message = start_tunnel("vmnode852", tmp_path)
        assert success is False
        assert "not found" in message.lower()

    @patch("subprocess.run")
    @patch("harness.tunnel_preflight.check_tunnel_available")
    def test_successful_startup(self, mock_check, mock_run, tmp_path):
        """Successful tunnel startup should return True."""
        # Create mock script
        script_path = tmp_path / "scripts" / "start-winrm-tunnels.sh"
        script_path.parent.mkdir(parents=True)
        script_path.write_text("#!/bin/bash\necho 'Started'")

        mock_run.return_value = MagicMock(returncode=0, stdout="Tunnel started", stderr="")
        mock_check.return_value = True

        success, message = start_tunnel("vmnode852", tmp_path)

        assert success is True
        assert "started" in message.lower()
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_script_failure(self, mock_run, tmp_path):
        """Script failure should return False with error message."""
        script_path = tmp_path / "scripts" / "start-winrm-tunnels.sh"
        script_path.parent.mkdir(parents=True)
        script_path.write_text("#!/bin/bash\nexit 1")

        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Connection refused")

        success, message = start_tunnel("vmnode852", tmp_path)

        assert success is False
        assert "Connection refused" in message or "failed" in message.lower()

    @patch("subprocess.run")
    def test_script_timeout(self, mock_run, tmp_path):
        """Script timeout should return False."""
        import subprocess

        script_path = tmp_path / "scripts" / "start-winrm-tunnels.sh"
        script_path.parent.mkdir(parents=True)
        script_path.write_text("#!/bin/bash\nsleep 100")

        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["bash"], timeout=10)

        success, message = start_tunnel("vmnode852", tmp_path, timeout=10)

        assert success is False
        assert "timed out" in message.lower()


class TestStopTunnel:
    """Tests for stop_tunnel function."""

    def test_non_tunneled_host_fails(self, tmp_path):
        """Non-tunneled hosts should fail immediately."""
        success, message = stop_tunnel("vmnode520", tmp_path)
        assert success is False
        assert "not a tunneled host" in message

    @patch("subprocess.run")
    def test_successful_stop(self, mock_run, tmp_path):
        """Successful tunnel stop should return True."""
        script_path = tmp_path / "scripts" / "start-winrm-tunnels.sh"
        script_path.parent.mkdir(parents=True)
        script_path.write_text("#!/bin/bash\necho 'Stopped'")

        mock_run.return_value = MagicMock(returncode=0)

        success, message = stop_tunnel("vmnode852", tmp_path)

        assert success is True
        assert "stopped" in message.lower()


class TestGetTunnelStatus:
    """Tests for get_tunnel_status function."""

    def test_non_tunneled_host(self):
        """Non-tunneled host should show no tunnel required."""
        status = get_tunnel_status("vmnode520")
        assert status["requires_tunnel"] is False
        assert status["tunnel_port"] is None
        assert status["tunnel_available"] is True  # True because not needed

    @patch("harness.tunnel_preflight.check_tunnel_available")
    def test_tunneled_host_available(self, mock_check):
        """Tunneled host with active tunnel should show available."""
        mock_check.return_value = True
        status = get_tunnel_status("vmnode852")

        assert status["requires_tunnel"] is True
        assert status["tunnel_port"] == 15852
        assert status["tunnel_available"] is True

    @patch("harness.tunnel_preflight.check_tunnel_available")
    def test_tunneled_host_unavailable(self, mock_check):
        """Tunneled host without active tunnel should show unavailable."""
        mock_check.return_value = False
        status = get_tunnel_status("vmnode876")

        assert status["requires_tunnel"] is True
        assert status["tunnel_port"] == 15876
        assert status["tunnel_available"] is False


class TestEnsureTunnelConnectivity:
    """Tests for ensure_tunnel_connectivity function."""

    def test_non_tunneled_host_succeeds(self, tmp_path):
        """Non-tunneled hosts should succeed immediately."""
        success, message = ensure_tunnel_connectivity("vmnode520", tmp_path)
        assert success is True
        assert "does not require tunneling" in message

    @patch("harness.tunnel_preflight.check_tunnel_available")
    def test_tunnel_already_running(self, mock_check, tmp_path):
        """Should succeed if tunnel is already running."""
        mock_check.return_value = True

        success, message = ensure_tunnel_connectivity("vmnode852", tmp_path)

        assert success is True
        assert "active" in message.lower()

    @patch("harness.tunnel_preflight.start_tunnel")
    @patch("harness.tunnel_preflight.check_tunnel_available")
    def test_auto_start_on_missing_tunnel(self, mock_check, mock_start, tmp_path):
        """Should attempt auto-start when tunnel not running."""
        mock_check.return_value = False
        mock_start.return_value = (True, "Tunnel started for vmnode876")

        success, message = ensure_tunnel_connectivity("vmnode876", tmp_path, auto_start=True)

        assert success is True
        mock_start.assert_called_once()

    @patch("harness.tunnel_preflight.check_tunnel_available")
    def test_auto_start_disabled(self, mock_check, tmp_path):
        """Should fail if tunnel not running and auto_start disabled."""
        mock_check.return_value = False

        success, message = ensure_tunnel_connectivity("vmnode852", tmp_path, auto_start=False)

        assert success is False
        assert "not running" in message.lower()
        assert "15852" in message

    @patch("harness.tunnel_preflight.start_tunnel")
    @patch("harness.tunnel_preflight.check_tunnel_available")
    def test_auto_start_failure(self, mock_check, mock_start, tmp_path):
        """Should fail if auto-start fails."""
        mock_check.return_value = False
        mock_start.return_value = (False, "Connection to xoxd-bates refused")

        success, message = ensure_tunnel_connectivity("vmnode876", tmp_path, auto_start=True)

        assert success is False
        assert "auto-start failed" in message.lower()


class TestErrorResolutionIntegration:
    """Test integration with error_resolution module."""

    def test_tunnel_error_classified_as_recoverable(self):
        """Tunnel errors should be classified as recoverable."""
        from harness.dag.error_resolution import ErrorType, classify_error

        state = {"role_name": "common", "deploy_target": "vmnode852"}
        error = classify_error(
            "Tunnel pre-flight failed: Tunnel not running for vmnode852",
            "run_molecule",
            state,
        )

        assert error.error_type == ErrorType.RECOVERABLE
        assert error.resolution_hint == "start_tunnel"

    def test_winrm_connection_refused_is_recoverable(self):
        """WinRM connection refused should be recoverable with tunnel hint."""
        from harness.dag.error_resolution import ErrorType, classify_error

        state = {"deploy_target": "vmnode876"}
        error = classify_error(
            "WinRM: connection refused to vmnode876",
            "run_molecule",
            state,
        )

        assert error.error_type == ErrorType.RECOVERABLE
        assert error.resolution_hint == "start_tunnel"

    def test_localhost_tunnel_port_timeout_is_recoverable(self):
        """Localhost tunnel port timeout should be recoverable."""
        from harness.dag.error_resolution import ErrorType, classify_error

        state = {"deploy_target": "vmnode851"}
        error = classify_error(
            "Connection to localhost port 15851 timed out",
            "run_molecule",
            state,
        )

        assert error.error_type == ErrorType.RECOVERABLE
        assert error.resolution_hint == "start_tunnel"


class TestConfigIntegration:
    """Test integration with harness config."""

    def test_testing_config_has_tunnel_fields(self):
        """TestingConfig should have tunnel fields."""
        from harness.config import TestingConfig

        config = TestingConfig()

        assert hasattr(config, "tunnel_auto_start")
        assert hasattr(config, "tunnel_startup_timeout")
        assert hasattr(config, "tunnel_script")

        # Check defaults
        assert config.tunnel_auto_start is True
        assert config.tunnel_startup_timeout == 30
        assert config.tunnel_script == "scripts/start-winrm-tunnels.sh"

    def test_testing_config_from_dict(self):
        """TestingConfig should load tunnel fields from dict."""
        from harness.config import TestingConfig

        config = TestingConfig(
            tunnel_auto_start=False,
            tunnel_startup_timeout=60,
            tunnel_script="custom/tunnel.sh",
        )

        assert config.tunnel_auto_start is False
        assert config.tunnel_startup_timeout == 60
        assert config.tunnel_script == "custom/tunnel.sh"
