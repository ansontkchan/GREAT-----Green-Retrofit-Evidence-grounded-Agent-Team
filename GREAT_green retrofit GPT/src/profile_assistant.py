from __future__ import annotations

from pathlib import Path

from .agents import UserProfile


def _ask(prompt: str, default: str | None = None) -> str:
    full = f"{prompt} [{default}]: " if default is not None else f"{prompt}: "
    ans = input(full).strip()
    return default if (not ans and default is not None) else ans


def _choose(label: str, options: list[str], default_index: int = 0) -> str:
    print(f"\n{label}:")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        choice = input(f"Choose 1-{len(options)} [{default_index + 1}]: ").strip() or str(default_index + 1)
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print(f"Please enter a number between 1 and {len(options)}.")


def collect_profile_cli() -> UserProfile:
    """CLI profile collection for local testing. Streamlit app provides the UI version."""
    print("\n=== GREAT User Profile Setup ===")
    print("We do NOT collect your name. Press Enter to accept defaults.\n")

    profession = _ask("Profession/role", "asset manager")
    organisation = _ask("Organisation (optional)", "")
    primary_concern = _ask("Primary concern", "energy and carbon")
    scope = _ask("Scope", "single office building")
    location = _ask("Location", "London, UK")
    timeframe = _ask("Timeframe / goal", "net-zero by 2040")

    horizon_str = _ask("Planning time horizon in years", "20")
    try:
        time_horizon_years = int(horizon_str)
    except ValueError:
        print("Invalid number; using 20 years.")
        time_horizon_years = 20

    breeam_choice = _choose(
        "BREEAM rating appetite",
        ["Outstanding", "Excellent", "Very Good", "Good", "Pass", "Unclassified", "No specific rating"],
        default_index=1,
    )
    standards_target = "No specific BREEAM rating target" if breeam_choice == "No specific rating" else f"BREEAM {breeam_choice}, WLCA-aligned"

    budget_band = _choose(
        "Budget band",
        [
            "low – essential / quick-payback only",
            "medium – balanced fabric + systems package",
            "high – deep retrofit / major capex acceptable",
        ],
        default_index=1,
    ).split(" – ")[0]

    risk_appetite = _choose(
        "Risk appetite",
        ["low – proven low-risk measures", "moderate – balanced ambition and risk", "high – willing to test innovative measures"],
        default_index=1,
    ).split(" – ")[0]

    extra_constraints = _ask("Extra constraints", "occupied during works")

    profile = UserProfile(
        profession=profession,
        organisation=organisation,
        primary_concern=primary_concern,
        scope=scope,
        location=location,
        timeframe=timeframe,
        time_horizon_years=time_horizon_years,
        standards_target=standards_target,
        budget_band=budget_band,
        risk_appetite=risk_appetite,
        extra_constraints=extra_constraints,
    )

    chat_dir = Path("chat_history")
    chat_dir.mkdir(exist_ok=True)
    (chat_dir / "user_profile.txt").write_text(profile.to_text(), encoding="utf-8")

    print("\n=== Profile summary ===")
    print(profile.to_text())
    confirm = _ask("Does this look correct? (y/n)", "y").lower()
    if confirm not in {"y", "yes"}:
        return collect_profile_cli()
    return profile
