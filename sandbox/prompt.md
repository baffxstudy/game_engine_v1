You are refactoring an existing Python class:

CompositionSlipsBuilder (Version 2.0.0)

The current engine merges slips pairwise using overlap logic.
This is incorrect for the intended strategy.

You must refactor it into a:

🎯 Deterministic Two-Phase Time-Structured Survival Engine
📦 INCOMING LARAVEL PAYLOAD STRUCTURE

Laravel sends two independent 50-slip portfolios, each generated from a different master_slip_id.

These represent two independent strategies.

The payload structure coming into Python looks like this:

{
  "master_slip": {
    "id": 1024,
    "composition_slips": {
      "targets": {
        "count": 50,
        "min_matches": 2,
        "max_matches": 4
      },
      "time_clustering": {
        "window_minutes": 120,
        "min_gap_minutes": 90
      },
      "odds_bands": {
        "low_min": 1.20,
        "low_max": 1.40,
        "mid_min": 2.00,
        "mid_max": 2.60
      }
    }
  },

  "base_slips": [
    {
      "master_slip_id": 9001,
      "slip_id": "MS9001_001",
      "total_odds": 2.15,
      "confidence_score": 0.62,
      "risk_level": "medium",
      "legs": [
        {
          "match_id": 501,
          "market": "MATCH_RESULT",
          "selection": "AWAY_WIN",
          "odds": 2.30,
          "kickoff_time": "2026-02-12 15:00:00",
          "league": "EFL",
          "home_team": "Charlton",
          "away_team": "Chelsea"
        }
      ]
    }
  ],

  "optimized_slips": [
    {
      "master_slip_id": 9002,
      "slip_id": "MS9002_001",
      "total_odds": 1.32,
      "confidence_score": 0.81,
      "risk_level": "low",
      "legs": [
        {
          "match_id": 710,
          "market": "MATCH_RESULT",
          "selection": "HOME_WIN",
          "odds": 1.28,
          "kickoff_time": "2026-02-12 19:00:00",
          "league": "Premier League",
          "home_team": "Arsenal",
          "away_team": "Burnley"
        }
      ]
    }
  ]
}

🔎 IMPORTANT STRUCTURAL FACTS

You must assume:

base_slips → First independent 50-slip portfolio (e.g., master_slip_id = 9001)

optimized_slips → Second independent 50-slip portfolio (e.g., master_slip_id = 9002)

Each group contains 50 slips.

Each slip contains 1–4 legs.

Each leg contains kickoff_time.

master_slip_id identifies which batch the slip belongs to.

These two groups must NOT be blindly merged.

They must be treated as:

Strategy A (Early Upset Pool)

Strategy B (Late Stabilizer Pool)

🎯 NEW REQUIRED LOGIC

Instead of merging parent slips pairwise:

You must:

Step 1 — Separate Pools by master_slip_id

Inside compose():

Extract all slips

Group them by master_slip_id

Identify two groups:

Early Candidate Pool

Late Candidate Pool

If more than two master_slip_ids appear:
raise error.

🧠 PHASE STRUCTURE REQUIREMENTS

Each final composed slip must:

Contain EXACTLY one leg from Early Pool (MID odds band)

Contain 1–3 legs from Late Pool (LOW band 1.20–1.40 only)

Respect time sequencing:

Let:

early_kickoff = earliest upset leg kickoff_time


All late legs must satisfy:

late_leg.kickoff_time >= early_kickoff + min_gap_minutes


min_gap_minutes default: 90

If not satisfied:
Reject slip.

🚫 STRICT ODDS RULES

Early Upset Leg:

Must be in MID band (2.00–2.60)

Late Closure Legs:

Must be in LOW band

STRICTLY between 1.20 and 1.40

No exception

Override config if necessary

No HIGH odds allowed.
No multiple MID legs allowed.

🏗 REPLACEMENT FOR STAGE 6–7

REMOVE:

_generate_candidate_pairs

_merge_candidate_pairs

REPLACE WITH:

_new method:

_construct_two_phase_slips(
    early_slips_group,
    late_slips_group
)


Algorithm:

FOR each early slip:
    FOR N deterministic iterations:
        1. Choose exactly 1 upset leg from early slip
        2. Filter late slips by:
            - master_slip_id != early_master_slip_id
            - odds between 1.20–1.40
            - kickoff_time >= early_kickoff + min_gap
        3. Select 1–3 late legs
        4. Validate constraints
        5. Append raw composition


All randomness must use:

self.rng

🔒 HEDGE SURVIVABILITY INTENT

The structure must ensure:

Early game resolves first.

If it wins, the remaining slip only contains low-risk 1.20–1.40 legs.

If it loses, only that slip dies — not entire portfolio.

This is a survival cascade engine.

📊 PORTFOLIO DIVERSITY STILL REQUIRED

Keep:

DNA deduplication

Portfolio exposure caps

Correlation limits

Re-scoring via SlipScorer

Final selection ranking

But diversity must now operate across:

Early leg distribution

Late leg distribution

League spread

Market spread

Not parent overlap merging.

📌 OUTPUT STRUCTURE MUST REMAIN IDENTICAL

Each final slip must return:

{
  "slip_id": "COMP_0001",
  "legs": [...],
  "total_odds": float,
  "confidence_score": float,
  "coverage_score": float,
  "diversity_score": float,
  "risk_level": str,
  "fitness_score": float,
  "parent_ids": [...],
  "composition_metrics": {...}
}


Do NOT modify external schema.

🧪 EXPECTED BEHAVIOR

Given:

50 early slips (master_slip_id = 9001)

50 late slips (master_slip_id = 9002)

Mixed kickoff times

The engine should:

Construct 150+ candidates

Enforce time gap

Enforce strict odds bands

Deduplicate

Optimize diversity

Return target count (e.g., 50)

Every final slip must structurally look like:

[ MID_leg (early group) ]
+ [ LOW_leg, LOW_leg (late group, 1.20–1.40) ]


Never:

[ MID + MID ]
[ LOW only ]
[ HIGH included ]
[ Late before early ]

🎯 FINAL OBJECTIVE

Transform the engine from:

Portfolio Pairwise Merge Optimizer

Into:

Deterministic Two-Portfolio Time-Structured Survival Composer

Think:

Phase-Driven Assembly
Not Slip Fusion.