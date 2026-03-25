"""CLI entrypoint - run the APM agent on the default project dataset."""

import json

from src.apm.agent import DATA_DIR, make_graph


def _load_project_context() -> str:
    """Read project files and format them as inline context for the user message."""
    files = {
        "meeting_minutes": DATA_DIR / "meeting_minutes_alpha.txt",
        "firm_sop": DATA_DIR / "firm_sop_v3.json",
        "system_logs": DATA_DIR / "system_logs.log",
    }
    sections = []
    for label, path in files.items():
        content = path.read_text()
        if path.suffix == ".json":
            content = json.dumps(json.loads(content), indent=2)
        sections.append(f"=== {label} ({path.name}) ===\n{content}")
    return "\n\n".join(sections)


def main() -> None:
    graph = make_graph()
    project_context = _load_project_context()

    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Analyse the following project files for compliance. "
                        "Cross-check against the Dubai Building Code and "
                        "produce the JSON risk report.\n\n"
                        f"{project_context}"
                    ),
                }
            ]
        },
    )

    # Print the final agent message
    final_msg = result["messages"][-1]
    print("\n=== APM Risk Report ===\n")
    print(final_msg.content)


if __name__ == "__main__":
    main()
