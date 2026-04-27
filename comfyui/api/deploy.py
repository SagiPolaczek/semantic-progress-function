#!/usr/bin/env python3
"""Submit the ReTime workflow to a ComfyUI server and wait for completion."""

import argparse
import json
import urllib.request
import uuid
import websocket
from pathlib import Path

WORKFLOW_PATH = Path(__file__).parent / "retime_workflow_api.json"
DEFAULT_PORT = 8188


def submit_workflow(workflow: dict, server: str, client_id: str) -> str:
    body = json.dumps({"prompt": workflow, "client_id": client_id}).encode()
    req = urllib.request.Request(
        f"{server}/prompt", data=body,
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req).read())
    if "error" in resp:
        raise RuntimeError(f"Server rejected workflow: {resp['error']}")
    return resp["prompt_id"]


def wait_for_completion(server: str, client_id: str, prompt_id: str):
    ws_url = server.replace("http://", "ws://").replace("https://", "wss://")
    ws = websocket.WebSocket()
    ws.connect(f"{ws_url}/ws?clientId={client_id}")
    try:
        while True:
            msg = json.loads(ws.recv())
            if msg["type"] == "progress":
                d = msg["data"]
                print(f"  step {d['value']}/{d['max']}", end="\r")
            elif msg["type"] == "executing":
                node = msg["data"].get("node")
                if node is None:
                    print("\nExecution finished.")
                    break
                title = msg["data"].get("display_node", node)
                print(f"Executing node {title}")
            elif msg["type"] == "execution_error":
                raise RuntimeError(msg["data"]["exception_message"])
    finally:
        ws.close()


def get_outputs(server: str, prompt_id: str) -> dict:
    resp = urllib.request.urlopen(f"{server}/history/{prompt_id}")
    history = json.loads(resp.read())
    return history.get(prompt_id, {}).get("outputs", {})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"ComfyUI server port (default: {DEFAULT_PORT})")
    parser.add_argument("--host", default="127.0.0.1",
                        help="ComfyUI server host (default: 127.0.0.1)")
    parser.add_argument("--workflow", type=Path, default=WORKFLOW_PATH,
                        help="Workflow JSON path (default: retime_workflow_api.json)")
    args = parser.parse_args()

    server = f"http://{args.host}:{args.port}"
    client_id = str(uuid.uuid4())

    workflow = json.loads(args.workflow.read_text())
    print(f"Loaded workflow: {args.workflow.name} ({len(workflow)} nodes)")
    print(f"Submitting to {server} ...")

    prompt_id = submit_workflow(workflow, server, client_id)
    print(f"Queued: {prompt_id}")

    wait_for_completion(server, client_id, prompt_id)

    outputs = get_outputs(server, prompt_id)
    for node_id, out in outputs.items():
        for key, files in out.items():
            if isinstance(files, list):
                for f in files:
                    if isinstance(f, dict) and "filename" in f:
                        print(f"  [{node_id}] {f['filename']}")


if __name__ == "__main__":
    main()
