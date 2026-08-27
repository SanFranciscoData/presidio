"""Unix/Linux container operations for DockerEnvironment."""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from presidio.environments.docker.docker import DockerEnvironment


class UnixOps:
    """File transfer and exec operations for Linux containers."""

    def __init__(self, env: DockerEnvironment) -> None:
        self._env = env

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        await self._env._run_docker_compose_command(
            ["cp", str(source_path), f"main:{target_path}"],
            check=True,
        )

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        await self._env._run_docker_compose_command(
            ["cp", f"{source_dir}/.", f"main:{target_dir}"],
            check=True,
        )
        # Fix CRLF line endings when the host is Windows: shell scripts with
        # Windows line endings fail to execute inside the Linux container.
        if sys.platform == "win32":
            await self._env._run_docker_compose_command(
                [
                    "exec",
                    "main",
                    "bash",
                    "-c",
                    f"find {target_dir} -type f \\( -name '*.sh' -o -name '*.py' "
                    "-o -name '*.ps1' -o -name '*.cmd' -o -name '*.bat' \\) "
                    "-exec sed -i 's/\\r$//' {} \\;",
                ],
                check=False,
            )

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await self._env._chown_to_host_user(source_path)
        await self._env._run_docker_compose_command(
            ["cp", f"main:{source_path}", str(target_path)],
            check=True,
        )

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        await self._env._chown_to_host_user(source_dir, recursive=True)
        await self._env._run_docker_compose_command(
            ["cp", f"main:{source_dir}/.", str(target_dir)],
            check=True,
        )

    async def download_file_from_service(
        self,
        service: str,
        source_path: str,
        target_path: Path | str,
    ) -> None:
        await self._chown_in_service(service, source_path)
        await self._env._run_docker_compose_command(
            ["cp", f"{service}:{source_path}", str(target_path)],
            check=True,
        )

    async def download_dir_from_service(
        self,
        service: str,
        source_dir: str,
        target_dir: Path | str,
    ) -> None:
        await self._chown_in_service(service, source_dir, recursive=True)
        await self._env._run_docker_compose_command(
            ["cp", f"{service}:{source_dir}/.", str(target_dir)],
            check=True,
        )

    async def _chown_in_service(
        self,
        service: str,
        path: str,
        recursive: bool = False,
    ) -> None:
        """Best-effort chown of a service container path to the host user."""
        if not hasattr(os, "getuid"):
            return
        flag = "-R " if recursive else ""
        await self._env._exec_in_compose_service(
            service,
            f"chown {flag}{os.getuid()}:{os.getgid()} {shlex.quote(path)}",
            user="root",
        )

    @staticmethod
    def exec_shell_args(command: str) -> list[str]:
        """Return the shell wrapper for executing *command* in a Linux container."""
        return ["bash", "-c", command]
