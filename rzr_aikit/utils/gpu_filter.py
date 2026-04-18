#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import csv
import sys
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GPUInfo:
    """GPU information from discovery JSON."""

    idx: int
    name: str
    ip: str
    pci_bus_id: Optional[str]
    uuid: str
    memory: Optional[dict]

    @classmethod
    def from_dict(cls, data: dict) -> "GPUInfo":
        """Create GPUInfo from discovery JSON."""
        return cls(
            idx=data.get("idx", 0),
            name=data.get("name", ""),
            ip=data.get("ip", ""),
            pci_bus_id=data.get("pci_bus_id"),
            uuid=data.get("uuid", ""),
            memory=data.get("memory"),
        )


def load_gpus(gpus_path: str) -> List[GPUInfo]:
    """Load GPU information from discovery JSON file."""
    try:
        with open(gpus_path, "r") as f:
            gpu_list = json.load(f)
    except Exception as e:
        print(f"Error: Failed to read {gpus_path}: {e}", file=sys.stderr)
        print("[info] Please run 'gpu-discover' first to generate the GPU information file.", file=sys.stderr)
        sys.exit(1)

    if not isinstance(gpu_list, list) or not gpu_list:
        print(f"Error: {gpus_path} is empty/invalid", file=sys.stderr)
        sys.exit(1)

    return [GPUInfo.from_dict(gpu_data) for gpu_data in gpu_list]


def create_parking_map(
    gpus: List[GPUInfo], selected_indices: List[int]
) -> Dict[str, List[int]]:
    """Create a map of IP -> [local_gpu_indices_to_park].
    
    Maps global GPU indices (from gpu-discover) to local GPU indices per node.
    The mapping accounts for vLLM's GPU ordering where GPUs are already sorted
    by PCI bus ID within each node.
    
    Args:
        gpus: List of GPUInfo objects in vLLM-ordered sequence from gpu-discover
        selected_indices: Global GPU indices to keep (not park)
    
    Returns:
        Dictionary mapping node IP to list of local GPU indices to park
    """
    if not selected_indices:
        print("Error: No GPUs specified for selection.", file=sys.stderr)
        sys.exit(1)

    # Validate selected indices
    for idx in selected_indices:
        if idx < 0 or idx >= len(gpus):
            print(
                f"Error: Selected index {idx} is out of range (0-{len(gpus)-1})",
                file=sys.stderr,
            )
            sys.exit(1)

    # Group GPUs by IP, preserving discovery order
    # The GPUs from gpu-discover are already in vLLM order (head first, then sorted)
    ip_to_gpus: Dict[str, List[Tuple[int, int]]] = {}

    for gpu in gpus:
        ip = gpu.ip
        if ip not in ip_to_gpus:
            ip_to_gpus[ip] = []
        # Local index is the position within this node's GPU list
        local_idx = len(ip_to_gpus[ip])
        ip_to_gpus[ip].append((gpu.idx, local_idx))

    # Determine which GPUs to keep (not park)
    selected_global_indices = set(selected_indices)

    # Create parking map: IP -> [local_indices_to_park]
    parking_map: Dict[str, List[int]] = {}

    for ip, gpu_list in ip_to_gpus.items():
        local_indices_to_park = []

        for global_idx, local_idx in gpu_list:
            if global_idx not in selected_global_indices:
                local_indices_to_park.append(local_idx)

        parking_map[ip] = sorted(local_indices_to_park)

    return parking_map


def park_gpus_ray(
    parking_map: Dict[str, List[int]], namespace: str = "parking"
) -> bool:
    """Park GPUs using Ray placement groups."""
    try:
        import ray
        from ray.util.placement_group import placement_group, remove_placement_group
    except ImportError:
        print(
            "Error: Ray not available. Install ray to use distributed parking.",
            file=sys.stderr,
        )
        return False

    try:
        ray.init(address="auto", namespace=namespace, ignore_reinit_error=True)
    except Exception as e:
        print(f"Error: Failed to connect to Ray cluster: {e}", file=sys.stderr)
        return False

    # Get available nodes
    nodes = [
        n for n in ray.nodes() if n.get("Alive") and n["Resources"].get("GPU", 0) > 0
    ]
    if not nodes:
        print("Error: No Ray nodes with GPUs detected.", file=sys.stderr)
        return False

    def pg_name(ip: str, local_idx: int) -> str:
        return f"park_{ip.replace('.', '_')}_gpu_{local_idx}"

    created_placements = {}

    # Create placement groups for all GPUs first
    for node in nodes:
        if not node["Alive"]:
            continue

        ip = node["NodeManagerAddress"]
        resources = node["Resources"]
        gpu_count = int(resources.get("GPU", 0))

        if gpu_count <= 0 or f"node:{ip}" not in resources:
            continue

        # Create placement groups for all GPUs on this node
        for local_idx in range(gpu_count):
            name = pg_name(ip, local_idx)

            # Remove any existing placement group with this name
            try:
                old_pg = ray.util.get_placement_group(name)
                if old_pg:
                    remove_placement_group(old_pg)
            except Exception:
                pass

            # Create new placement group
            try:
                pg = placement_group(
                    bundles=[{"GPU": 1, "CPU": 0.1, f"node:{ip}": 0.001}],
                    strategy="PACK",
                    name=name,
                    lifetime="detached",
                )
                created_placements[(ip, local_idx)] = pg
            except Exception as e:
                print(
                    f"Warning: Failed to create placement group for {ip}:{local_idx}: {e}",
                    file=sys.stderr,
                )

    # Wait for all placement groups to be ready
    if created_placements:
        try:
            ray.get([pg.ready() for pg in created_placements.values()])
        except Exception as e:
            print(
                f"Warning: Some placement groups failed to become ready: {e}",
                file=sys.stderr,
            )

    # Remove placement groups for GPUs that should NOT be parked
    requested_parks = {
        (ip, local_idx)
        for ip, local_indices in parking_map.items()
        for local_idx in local_indices
    }

    for (ip, local_idx), pg in created_placements.items():
        if (ip, local_idx) not in requested_parks:
            try:
                remove_placement_group(pg)
            except Exception as e:
                print(
                    f"Warning: Failed to remove placement group for {ip}:{local_idx}: {e}",
                    file=sys.stderr,
                )

    return True


def main_distributed(
    selected_indices: List[int], gpus: List[GPUInfo], namespace: str
) -> bool:
    """Main logic for distributed GPU parking."""
    # Create parking map
    parking_map = create_parking_map(gpus, selected_indices)

    # Apply parking
    return park_gpus_ray(parking_map, namespace)


def main(args: argparse.Namespace):
    """Main parking logic."""
    # Parse selected GPU indices
    try:
        selected_indices = [
            int(x.strip()) for x in args.selected_gpus.split(",") if x.strip()
        ]
    except ValueError:
        print(
            "Error: Invalid selected GPUs format. Use comma-separated integers, e.g. '1,2,5'",
            file=sys.stderr,
        )
        sys.exit(1)

    if not selected_indices:
        print("Error: No GPUs specified for selection.", file=sys.stderr)
        sys.exit(1)

    gpus = load_gpus(args.gpus_path)
    main_distributed(selected_indices, gpus, args.namespace)

    # print selected GPUs
    print("Selected GPUs:")
    for idx in sorted(selected_indices):
        if idx < len(gpus):
            gpu = gpus[idx]

            print(f"[{idx}] - {gpu.name}")


def cli():
    """Command-line interface for GPU parking tool."""
    parser = argparse.ArgumentParser(
        prog="gpu-park",
        description="Park unused GPUs to reserve selected ones for inference",
    )

    parser.add_argument(
        "selected_gpus",
        help="Comma-separated list of GPU indices to keep for inference (e.g., '0,1,3')",
    )

    # Distributed-specific options
    parser.add_argument(
        "--gpus-path",
        default=Path.home() / ".cache" / "gpus.json",
        help="Path to GPU discovery JSON file for distributed mode (default: ~/.cache/gpus.json)",
    )

    parser.add_argument(
        "--namespace",
        default="parking",
        help="Ray namespace for placement groups in distributed mode (default: parking)",
    )

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli()
