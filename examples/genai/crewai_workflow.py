"""Optional CrewAI example with bounded, review-oriented responsibilities."""
from __future__ import annotations


def main() -> None:
    from crewai import Agent, Crew, Process, Task

    analyst = Agent(
        role="Data quality analyst",
        goal="Identify documented data-quality risks without inventing evidence",
        backstory="You review pipeline reports and make conservative recommendations.",
        verbose=False,
    )
    reviewer = Agent(
        role="Model risk reviewer",
        goal="Check whether conclusions match the available evaluation metrics",
        backstory="You challenge unsupported claims and require traceable evidence.",
        verbose=False,
    )
    analysis = Task(description="Review artifacts/reports/data-quality.json and summarize risks.", agent=analyst)
    review = Task(description="Review the analyst summary against model-metrics.json.", agent=reviewer)
    crew = Crew(agents=[analyst, reviewer], tasks=[analysis, review], process=Process.sequential)
    print(crew.kickoff())


if __name__ == "__main__":
    main()
